import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from saveNCfile import savenc
from saveNCfile_for_activations import savenc_for_activations
from data_loader import load_test_data
from data_loader import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage


### PATHS and FLAGS ###

path_static_activations = '/glade/scratch/asheshc/theory-interp/QG/Base_Simulations/U-NET_no_sponge/activations_analysis/'
path_weights = '/glade/scratch/asheshc/theory-interp/QG/Base_Simulations/U-NET_no_sponge/weights_analysis/'

path_outputs = '/glade/scratch/asheshc/theory-interp/QG/Base_Simulations/U-NET_no_sponge/RK4_UNET_FFT_loss_outputs/'



FLAGS_WEIGHTS_DUMP=0
FLAGS_ACTIVATIONS_DUMP=0







##### prepare test data ###################################################

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/set'+str(9)+'/PSI_output.nc')
lat=np.asarray(FF['lat'])
lon=np.asarray(FF['lon'])

lead = 1

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr,psi_test_mean_torch = load_test_data(FF,lead)

dim_lon = int(np.size(psi_test_label_Tr,2))
new_lat = lat[24:166]
lim=0.4



############################################## Low Pass function ##########################
def lowpass_torch(input, limit):
    pass1 = torch.abs(torch.fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(torch.fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    fft_input = torch.fft.rfft2(input)
    return torch.fft.irfft2(fft_input.cuda() * kernel.cuda(), s=input.shape[-2:])


#########################################






###############################################################################
#################### Remove Large Scales #####################################
psi_test_input_Tr_torch_small_scale =  torch.empty((psi_test_input_Tr_torch.shape[0],2,dim_lon,96))
psi_test_label_Tr_torch_small_scale =  torch.empty((psi_test_label_Tr_torch.shape[0],2,dim_lon,96))

for batches in range(0,psi_test_input_Tr_torch.shape[0]):

         psi_test_input_Tr_torch_large1 = (lowpass_torch(psi_test_input_Tr_torch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])
         psi_test_input_Tr_torch_large2 = (lowpass_torch(psi_test_input_Tr_torch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])

         psi_test_label_Tr_torch_large1 = (lowpass_torch(psi_test_label_Tr_torch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])
         psi_test_label_Tr_torch_large2 = (lowpass_torch(psi_test_label_Tr_torch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])

         psi_test_input_Tr_torch_large = torch.cat((psi_test_input_Tr_torch_large1,psi_test_input_Tr_torch_large2),1)
         psi_test_label_Tr_torch_large = torch.cat((psi_test_label_Tr_torch_large1,psi_test_label_Tr_torch_large2),1)

         psi_test_input_Tr_torch_small_scale[batches,:,:,:] = psi_test_input_Tr_torch[batches,:,:,:].float().cuda() - psi_test_input_Tr_torch_large[0,:,:,:].float().cuda()
         psi_test_label_Tr_torch_small_scale[batches,:,:,:] = psi_test_label_Tr_torch[batches,:,:,:].float().cuda() - psi_test_label_Tr_torch_large[0,:,:,:].float().cuda()



################### Load training data files ########################################
fileList_train=[]
mylist = [1,2,3,4,5,6,7,8]
for k in mylist:
  fileList_train.append ('/glade/scratch/asheshc/theory-interp/QG/set'+str(k)+'/PSI_output.nc')
##########################################################################################

def store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training,epoch,out,x1,x2,x3,x4,x5,x6):

   Act_encoder[epoch,0,:,:,:,:] = x1.detach().cpu().numpy()
   Act_encoder[epoch,1,:,:,:,:] = x2.detach().cpu().numpy()
   Act_encoder[epoch,2,:,:,:,:] = x3.detach().cpu().numpy()
   Act_encoder[epoch,3,:,:,:,:] = x4.detach().cpu().numpy()

   Act_decoder1[epoch,:,:,:,:] = x5.detach().cpu().numpy()
   Act_decoder2[epoch,:,:,:,:] = x6.detach().cpu().numpy()





   output_training [epoch,:,:,:,:] = out.detach().cpu().numpy()

   return Act_encoder, Act_decoder1, Act_decoder2, output_training

def store_weights (net,epoch,hidden_weights_encoder,hidden_weights_decoder1,final_weights_network):

  hidden_weights_encoder[epoch,0,:,:,:,:] = net.hidden1.weight.data.cpu()
  hidden_weights_encoder[epoch,1,:,:,:,:] = net.hidden2.weight.data.cpu()
  hidden_weights_encoder[epoch,2,:,:,:,:] = net.hidden3.weight.data.cpu()
  hidden_weights_encoder[epoch,3,:,:,:,:] = net.hidden4.weight.data.cpu()


  hidden_weights_decoder1[epoch,:,:,:,:] = net.hidden5.weight.data.cpu()
  final_weights_network[epoch,:,:,:,:] = net.hidden6.weight.data.cpu()

  return hidden_weights_encoder, hidden_weights_decoder1, final_weights_network



def my_loss(output, target,wavenum_init,lamda_reg):

 loss1 = torch.mean((output-target)**2)

 out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
 target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

 loss2 = torch.mean(torch.abs(out_fft[:,0,wavenum_init:]-target_fft[:,0,wavenum_init:]))
 loss3 = torch.mean(torch.abs(out_fft[:,1,wavenum_init:]-target_fft[:,1,wavenum_init:]))


 loss = (1-lamda_reg)*loss1 + 0.5*lamda_reg*loss2 +0.5*lamda_reg*loss3

 return loss



def RK4step(net,input_batch):
 output_1,_,_,_,_,_,_ = net(input_batch.cuda())
 output_2,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_1)
 output_3,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_2)
 output_4,_,_,_,_,_,_ = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6


 
  


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = (nn.Conv2d(2, 64, kernel_size=5, stride=1, padding='same'))
        self.hidden1 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden2 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))


        self.hidden5 = (nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(192, 2, kernel_size=5, stride=1, padding='same' ))
    
    def forward (self,x):

        x1 = F.relu (self.input_layer(x))
        x2 = F.relu (self.hidden1(x1))
        x3 = F.relu (self.hidden2(x2))
        x4 = F.relu (self.hidden3(x3))

        x5 = torch.cat ((F.relu(self.hidden4(x4)),x3), dim =1)
        x6 = torch.cat ((F.relu(self.hidden5(x5)),x2), dim =1)
        

        out = (self.hidden6(x6))


        return out, x1, x2, x3, x4, x5, x6


net = CNN()

net.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('**** Number of Trainable Parameters in BNN')
count_parameters(net)


batch_size = 10
num_epochs = 15
num_samples = 2
trainN = 7000
lambda_reg =0.2
wavenum_init=30
lim=0.10
Act_encoder = np.zeros([num_epochs,4,num_samples,64,dim_lon,96])   #### Last three: number of channels, Nalt, Nlon
Act_decoder1 = np.zeros([num_epochs,num_samples,128,dim_lon,96])
Act_decoder2 = np.zeros([num_epochs,num_samples,192,dim_lon,96])
output_training = np.zeros([num_epochs,num_samples,2, dim_lon, 96])

hidden_weights_encoder = np.zeros([num_epochs,4,64,64,5,5])
hidden_weights_decoder1 = np.zeros([num_epochs,128,128,5,5])
final_weights_network = np.zeros([num_epochs,2,192,5,5])




for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for loop in fileList_train:
     print('Training loop index',loop)

     psi_input_Tr_torch, psi_label_Tr_torch = load_train_data(loop, lead, trainN)

     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_input_Tr_torch[indices,:,:,:], psi_label_Tr_torch[indices,:,:,:]
        input_batch_small_scale = torch.empty((batch_size,2,dim_lon,96))
        label_batch_small_scale = torch.empty((batch_size,2,dim_lon,96))



        for batches in range(0,batch_size):

         input_batch_large1 = (lowpass_torch(input_batch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])                   
         input_batch_large2 = (lowpass_torch(input_batch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])

         label_batch_large1 = (lowpass_torch(label_batch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])
         label_batch_large2 = (lowpass_torch(label_batch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])                            

         input_batch_large = torch.cat((input_batch_large1,input_batch_large2),1)
         label_batch_large = torch.cat((label_batch_large1,label_batch_large2),1)

         input_batch_small_scale[batches,:,:,:] = input_batch[batches,:,:,:].float().cuda() - input_batch_large[0,:,:,:].float().cuda() 
         label_batch_small_scale[batches,:,:,:] = label_batch[batches,:,:,:].float().cuda() - label_batch_large[0,:,:,:].float().cuda() 

        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
#        output,_,_,_,_,_,_ = net(input_batch.cuda())
        output = RK4step(net,input_batch.cuda())
        loss = my_loss(output, label_batch_small_scale.cuda(),wavenum_init,lambda_reg)
        loss.backward()
        optimizer.step()
        output_val = RK4step (net,psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,dim_lon,96]).cuda())
        val_loss = my_loss(output_val, psi_test_label_Tr_torch_small_scale[0:num_samples].reshape([num_samples,2,dim_lon,96]).cuda(),wavenum_init,lambda_reg)
        # print statistics

        if step % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, loss))
            print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, step + 1, val_loss))
            running_loss = 0.0
#    out = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,dim_lon,96]).cuda())

#    hidden_weights_encoder, hidden_weights_decoder1, final_weights_network = store_weights(net,epoch,hidden_weights_encoder, hidden_weights_decoder1, final_weights_network)

#    Act_encoder, Act_decoder1, Act_decoder2, output_training = store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training, epoch,out,x1,x2,x3,x4,x5,x6)

print('Finished Training')


torch.save(net.state_dict(), './BNN_RK4UNET_anomaly_full_scale_to_small_scale_FFT_loss_lead'+str(lead)+'.pt')

print('BNN Model Saved')

if (FLAGS_ACTIVATIONS_DUMP ==1):
 savenc_for_activations(Act_encoder, Act_decoder1, Act_decoder2,output_training,2,num_epochs,4,num_samples,64,128,192,dim_lon,96,path_static_activations+'BNN_UNET_no_sponge_FFT_loss_waveinit_'+str(wavenum_init)+'_Activations_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.nc')

 print('Saved Activations for BNN')

if (FLAGS_WEIGHTS_DUMP ==1):

 matfiledata = {}
 matfiledata[u'hidden_weights_encoder'] = hidden_weights_encoder
 matfiledata[u'hidden_weights_decoder'] = hidden_weights_decoder1
 matfiledata[u'final_layer_weights'] = final_weights_network
 hdf5storage.write(matfiledata, '.', path_weights+'BNN_RK4UNET_no_sponge_FFT_loss_waveinit_'+str(wavenum_init)+'Weights_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.mat', matlab_compatible=True)

 print('Saved Weights for BNN')

'''
############# Auto-regressive prediction #####################
M=1000
autoreg_pred = np.zeros([M,2,dim_lon,96])

for k in range(0,M):

  if (k==0):

    out = (RK4step(net,psi_test_input_Tr_torch_small_scale[k].reshape([1,2,dim_lon,96]).cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

  else:

    out = (RK4step(net,torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,dim_lon,96])).float().cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

savenc(autoreg_pred, lon, new_lat, path_outputs+'predicted_RK4_FFT_trained_on_small_scales_anomaly_waveinit_'+str(wavenum_init)+'_lead'+str(lead)+'.nc')
#savenc(psi_test_label_Tr, lon, new_lat, path_outputs+'truth_RK4_FFT_loss_waveinit_'+str(wavenum_init)+'lead'+str(lead)+'.nc')
'''
