# import libraries
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim

from models.seq2seq_ConvLSTM_Unet import EncoderDecoderConvLSTM
from data.load_dataset import Load_Data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=32, help='number of hidden dim for ConvLSTM layers')
parser.add_argument('--out_chan', type=int, default=3, help='output n_channels')
parser.add_argument('--in_chan', type=int, default=1, help='input n_channels')
parser.add_argument('--dataset_v', type=str, default='5', help='dataset_v_num')

opt = parser.parse_args()

##########################
######### MODEL ##########
##########################

class Precipitation_Lightning(pl.LightningModule):
    
    def __init__(self, hparams=None, model=None):
        super(Precipitation_Lightning, self).__init__()
        
        # default config
        self.model = model
        
        # Training config
        self.criterion = torch.nn.L1Loss()
        self.batch_size = opt.batch_size
        self.n_steps_past = 1
        self.n_steps_ahead = 1
        self.file_list = os.listdir('/home/yj/Computer_Vision/Ther2RGB-Translation/T2R_Dataset/train_A/')
        self.train_list, self.val_list = self.file_list[:2420], self.file_list[2421:]
        
    def forward(self, x):
        x = x.to(device='cuda')
        
        output = self.model(x, future_seq=self.n_steps_ahead)
        
        return output
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        x = x.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        loss = self.criterion(y_hat, y)
        
        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()
        
        tar, pred = deepcopy(y.permute(1,2,0).cpu().detach().numpy()*255), deepcopy(y_hat.permute(1,2,0).cpu().detach().numpy()*255)
        
        (score, diff) = ssim(tar, pred, full=True, multichannel=True)
        score = torch.from_numpy(np.array(score)).to(device='cuda')
        
        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}
        
        return {'loss': loss, 'log': tensorboard_logs, 'SSIM': score}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['SSIM'] for x in outputs]).mean()
        
        self.logger.experiment.add_scalars("Avg_Loss", {"Train_loss": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Avg_SSIM", {"Train_SSIM": avg_ssim}, self.current_epoch)
        
        return {'avg_loss': avg_loss, 'avg_ssim': avg_ssim}
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        x = x.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        
        loss = self.criterion(y_hat, y)
        tar, pred = deepcopy(y.permute(1,2,0).cpu().numpy()*255), deepcopy(y_hat.permute(1,2,0).cpu().numpy()*255)
        
        (score, diff) = ssim(tar, pred, full=True, multichannel=True)
        score = torch.from_numpy(np.array(score)).to(device='cuda')
        
        return {'val_loss': loss, 'SSIM': score}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['SSIM'] for x in outputs]).mean()
        
        self.logger.experiment.add_scalars("Avg_Loss", {"Val_loss": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Avg_SSIM", {"Val_SSIM": avg_ssim}, self.current_epoch)
        
        return {'avg_loss': avg_loss, 'avg_ssim': avg_ssim}
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2), weight_decay=0.0001)
        return optimizer
    
    @pl.data_loader
    def train_dataloader(self):
        train_data = Load_Data(
            file_list=self.train_list)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)
        
        return train_loader
    
    @pl.data_loader
    def val_dataloader(self):
        val_data = Load_Data(
            file_list=self.val_list)
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=True)
        
        return val_loader

def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=opt.in_chan, out_chan=opt.out_chan)
    
    model = Precipitation_Lightning(model=conv_lstm_model)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    
    trainer = Trainer(max_epochs=opt.epochs,
                      gpus=[0,1,2,3],
                      distributed_backend='dp',
                      early_stop_callback=False,
                      use_amp=opt.use_amp,
                      checkpoint_callback=checkpoint_callback)
    
    trainer.fit(model)


if __name__ == '__main__':
    p1 = Process(target=run_trainer)                    # start trainer
    p1.start()
    p1.join()
