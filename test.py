import cv2
import os
import torch
import numpy as np
from models.seq2seq_ConvLSTM_Unet import EncoderDecoderConvLSTM
from main import Precipitation_Lightning
from scipy import stats
import math


###############################################################
v_num = 1
in_chan = 1
out_chan = 3
ckpt = 74
###############################################################
data_path = '/home/yj/Computer_Vision/Ther2RGB-Translation/T2R_Dataset/test_A/'
file_list = os.listdir(data_path)

v_path = '/home/yj/Computer_Vision/Ther2RGB-Translation/lightning_logs/version_'+str(v_num)+'/checkpoints/'

result = []

#for n in range(len(file_list)):
for n in range(1):
    
    x = cv2.imread(data_path + file_list[n], cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(2)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x = x.permute(0, 1, 4, 2, 3)
    x = x/255*2-1
        
    conv_lstm_model = EncoderDecoderConvLSTM(nf=32, in_chan=in_chan, out_chan=out_chan)
    model_test = Precipitation_Lightning(model=conv_lstm_model)
    ckpt_path = v_path + 'epoch='+str(ckpt)+'.ckpt'
    
    checkpoint = torch.load(ckpt_path)
    model_test.load_state_dict(checkpoint['state_dict'])
    model_test = model_test.to(device='cuda')
    
    # x = x[:,:,3:]
    y_hat = model_test.forward(x).squeeze().permute(1,2,0).cpu().detach().numpy()
    
    result.append([y, y_hat])

result = np.stack(result, axis=0)
np.save('/home/yj/convLSTM/pytorch/Radar_Estimation/val_test/result', result)



#%%
    # version_22
    x = x[:,:,3:]
    # 

    # version_23
    x += 1
    x[:,:,:,0] = 2 - x[:,:,:,0]
    x[:,:,:,2] = 2 - x[:,:,:,2]
    x[:,:,:,5] = 2 - x[:,:,:,5]
    x -= 1
    # 