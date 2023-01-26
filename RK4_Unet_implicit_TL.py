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

path_outputs = '/glade/scratch/asheshc/theory-interp/QG/Base_Simulations/U-NET_no_sponge/RK4_UNET_TWO_NETWORK_MODEL/'



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


###############################################################################
def lowpass_torch(input, limit):
    pass1 = torch.abs(torch.fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(torch.fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    fft_input = torch.fft.rfft2(input)
    return torch.fft.irfft2(fft_input.cuda() * kernel.cuda(), s=input.shape[-2:])

##################### Remove Large Scales ###################################

psi_test_input_Tr_torch_small_scale =  torch.empty((psi_test_input_Tr_torch.shape[0],2,dim_lon,96))
psi_test_label_Tr_torch_small_scale =  torch.empty((psi_test_label_Tr_torch.shape[0],2,dim_lon,96))

lim=0.10



for batches in range(0,psi_test_input_Tr_torch.shape[0]):

         psi_test_input_Tr_torch_large1 = (lowpass_torch(psi_test_input_Tr_torch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])
         psi_test_input_Tr_torch_large2 = (lowpass_torch(psi_test_input_Tr_torch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])

         psi_test_label_Tr_torch_large1 = (lowpass_torch(psi_test_label_Tr_torch[batches,0,:,:],lim)).reshape([1,1,dim_lon,96])
         psi_test_label_Tr_torch_large2 = (lowpass_torch(psi_test_label_Tr_torch[batches,1,:,:],lim)).reshape([1,1,dim_lon,96])

         psi_test_input_Tr_torch_large = torch.cat((psi_test_input_Tr_torch_large1,psi_test_input_Tr_torch_large2),1)
         psi_test_label_Tr_torch_large = torch.cat((psi_test_label_Tr_torch_large1,psi_test_label_Tr_torch_large2),1)

         psi_test_input_Tr_torch_small_scale[batches,:,:,:] = psi_test_input_Tr_torch[batches,:,:,:].float().cuda() - psi_test_input_Tr_torch_large[0,:,:,:].float().cuda()
         psi_test_label_Tr_torch_small_scale[batches,:,:,:] = psi_test_label_Tr_torch[batches,:,:,:].float().cuda() - psi_test_label_Tr_torch_large[0,:,:,:].float().cuda()

####################################################################################



def my_loss(output, target,wavenum_init,lambda_reg):

 loss1 = torch.mean((output-target)**2)




 out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
 target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

 loss2 = torch.mean(torch.abs(out_fft[:,0,wavenum_init:]-target_fft[:,0,wavenum_init:]))
 
 loss3 = torch.mean(torch.abs(out_fft[:,1,wavenum_init:]-target_fft[:,1,wavenum_init:]))


 loss =  (1-lambda_reg)*loss1+0.5*lambda_reg*loss2 +0.5*lambda_reg*loss3

 return loss






def my_scaled_loss(output, target,wavenum_init,lambda_reg):

 loss1 = torch.mean((output-target)**2)




 out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
 target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

 loss2 = torch.mean(torch.log(torch.abs(out_fft[:,0,wavenum_init:]))-torch.log(torch.abs(target_fft[:,0,wavenum_init:])))

 loss3 = torch.mean(torch.log(torch.abs(out_fft[:,1,wavenum_init:]))-torch.log(torch.abs(target_fft[:,1,wavenum_init:]))) 

 loss =  (1-lambda_reg)*loss1+0.5*lambda_reg*loss2 +0.5*lambda_reg*loss3

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
net.load_state_dict(torch.load('BNN_RK4UNET_anomaly_FFT_loss_lead'+str(lead)+'.pt'))
net.cuda()
net.eval()



net_small_scale = CNN()
net_small_scale.load_state_dict(torch.load('BNN_UNET_anomaly_large_scale_to_small_scale_low_pass_0.1_FFT_loss_lead'+str(lead)+'.pt'))
net_small_scale.cuda()
net_small_scale.eval()



### Implicit layer as the last two layers ####

def set_net_small_scale (a):
 net_small_scale.input_layer.weight.requires_grad = False
 net_small_scale.hidden1.weight.requires_grad = False 
 net_small_scale.hidden2.weight.requires_grad = True
 net_small_scale.hidden3.weight.requires_grad = True
 net_small_scale.hidden4.weight.requires_grad = True
 net_small_scale.hidden5.weight.requires_grad = False
 net_small_scale.hidden6.weight.requires_grad = False

 net_small_scale.input_layer.bias.requires_grad = False
 net_small_scale.hidden1.bias.requires_grad = False
 net_small_scale.hidden2.bias.requires_grad = True
 net_small_scale.hidden3.bias.requires_grad = True
 net_small_scale.hidden4.bias.requires_grad = True
 net_small_scale.hidden5.bias.requires_grad = False
 net_small_scale.hidden6.bias.requires_grad = False


def set_net (a):
 net_small_scale.input_layer.weight.requires_grad = False
 net_small_scale.hidden1.weight.requires_grad = False
 net_small_scale.hidden2.weight.requires_grad = True
 net_small_scale.hidden3.weight.requires_grad = True
 net_small_scale.hidden4.weight.requires_grad = True
 net_small_scale.hidden5.weight.requires_grad = False
 net_small_scale.hidden6.weight.requires_grad = False

 net_small_scale.input_layer.bias.requires_grad = False
 net_small_scale.hidden1.bias.requires_grad = False
 net_small_scale.hidden2.bias.requires_grad = True
 net_small_scale.hidden3.bias.requires_grad = True
 net_small_scale.hidden4.bias.requires_grad = True
 net_small_scale.hidden5.bias.requires_grad = False
 net_small_scale.hidden6.bias.requires_grad = False




wavenumber_init =0
wavenumber_init_full_scale = 10
num_epochs = 1000
lambda_reg=0.99
limit=lim
############# Auto-regressive prediction #####################
M=1000
autoreg_pred = np.zeros([M,2,dim_lon,96])

for k in range(0,M):

  print('Number of Steps',k)
  if (k==0):

    out = (RK4step(net,psi_test_input_Tr_torch[k].reshape([1,2,dim_lon,96]).cuda()))
    
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

  else:

    if(k % 5 == 0):
     if (k > 5):
           
           net_small_scale.load_state_dict(torch.load('./IO_layer_multiple_layers_lead'+str(lead)+'.pt'))
           net_small_scale.cuda()
           net_small_scale.eval()
           set_net_small_scale(1)

           
           net.load_state_dict(torch.load('./IO_layer_full_scale_multiple_layers_lead'+str(lead)+'.pt'))
           net.cuda()
           net.eval()
           set_net(1)

 
     else:
        
           net_small_scale.load_state_dict(torch.load('BNN_UNET_anomaly_large_scale_to_small_scale_low_pass_0.1_FFT_loss_lead'+str(lead)+'.pt'))
           net_small_scale.cuda()
           net_small_scale.eval()
           set_net_small_scale(1) 

           net.load_state_dict(torch.load('BNN_RK4UNET_anomaly_FFT_loss_lead'+str(lead)+'.pt'))
           net.cuda()
           net.eval()
           set_net(1)        


     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
     for epoch in range(0,num_epochs):
      optimizer.zero_grad()
      out = RK4step(net, torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,dim_lon,96])).float().cuda())
      loss = my_loss(out,psi_test_input_Tr_torch[0].reshape([1,2,dim_lon,96]).cuda(),wavenumber_init_full_scale,lambda_reg)
      loss.backward(retain_graph=True)
      optimizer.step()
      print('epoch',epoch)
      print('Loss',loss)

     torch.save(net.state_dict(), './IO_layer_full_scale_multiple_layers_lead'+str(lead)+'.pt')     

     output = (RK4step(net,torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,dim_lon,96])).float().cuda()))
     output_large1 = (lowpass_torch(output[0,0,:,:],limit)).reshape([1,1,dim_lon,96])
     output_large2 = (lowpass_torch(output[0,1,:,:],limit)).reshape([1,1,dim_lon,96])
     output_large =torch.cat((output_large1,output_large2),1).reshape([1,2,dim_lon,96])

     initial_time_small1 = (psi_test_input_Tr_torch_small_scale[0,0,:,:]).float().cuda().reshape([1,1,dim_lon,96])-lowpass_torch((psi_test_input_Tr_torch_small_scale[0,0,:,:]).float().cuda(),limit).reshape([1,1,dim_lon,96])

     initial_time_small2 = (psi_test_input_Tr_torch_small_scale[0,1,:,:]).float().cuda().reshape([1,1,dim_lon,96])-lowpass_torch((psi_test_input_Tr_torch_small_scale[0,1,:,:]).float().cuda(),limit).reshape([1,1,dim_lon,96])

     initial_time_small = torch.cat((initial_time_small1,initial_time_small2),1).reshape([1,2,dim_lon,96])


     input_full_scale = torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,dim_lon,96])).float().cuda()   
     input_large1 = (lowpass_torch(input_full_scale[0,0,:,:],limit))
     input_large2 = (lowpass_torch(input_full_scale[0,1,:,:],limit))
     input_large = torch.cat((input_large1,input_large2),1).reshape([1,2,dim_lon,96])
      
     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net_small_scale.parameters()), lr=0.001, momentum=0.9)
     for epoch in range(0,num_epochs):
      optimizer.zero_grad()
      out,_,_,_,_,_,_ = net_small_scale(input_large.float().cuda()) 
      loss = my_loss (out,initial_time_small,wavenumber_init,lambda_reg)
      loss.backward(retain_graph=True)
      optimizer.step()
      print('epoch',epoch)
      print('Loss',loss)

     torch.save(net_small_scale.state_dict(), './IO_layer_multiple_layers_lead'+str(lead)+'.pt')
     out,_,_,_,_,_,_ = (net_small_scale(input_large.float().cuda()))
     out =(out+output_large).reshape([1,2,dim_lon,96])
    else:
     net = CNN()
     net.load_state_dict(torch.load('BNN_RK4UNET_anomaly_FFT_loss_lead'+str(lead)+'.pt'))
     net.cuda()
     net.eval()
     out = (RK4step(net,torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,dim_lon,96])).float().cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

savenc(autoreg_pred, lon, new_lat, path_outputs+'predicted_RK4_implicit_layer_two_network_model_large_to_small_with_anomaly_reusedTL_multiple_layers_every5_FFT_lambda_'+str(lambda_reg)+'_waveinit_full_scale'+str(wavenumber_init_full_scale)+'_lead'+str(lead)+'num_time_steps'+str(M)+'.nc')
#savenc(psi_test_label_Tr, lon, new_lat, path_outputs+'truth_RK4_FFT_loss_waveinit_'+str(wavenum_init)+'lead'+str(lead)+'.nc')

