# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:16:40 2020

@author: LENOVO
"""
import socket
import numpy as np
import os, datetime
import torch
import cv2

class Load_Data(object):
    
    def __init__(self, file_list=None):
        self.file_list = file_list
        self.path = '/home/yj/Computer_Vision/Ther2RGB-Translation/T2R_Dataset/'
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x = cv2.imread(self.path + 'train_A/' + self.file_list[index], cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(x).float()
        x = x.unsqueeze(2)
        x = x.unsqueeze(0)
        x = x/255*2-1
        
        y = cv2.imread(self.path + 'train_B/' + 'V' + self.file_list[index][1:])
        y = torch.from_numpy(y).float()
        y = y.unsqueeze(0)
        y = y/255*2-1

        return x,y

