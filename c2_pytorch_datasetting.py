#!/usr/bin/env Python
# coding=utf-8

import os 
import pickle
import pdb
import numpy as np

from c1_dataset_dividing import dataset_operator

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
# from torchvision import transforms, utils

class BSdataset(Dataset):
    def __init__(self, train=True, transform=None, obpath=None):
        # new = dataset_operator()
        if obpath is None:
            obpath = os.path.join(os.getcwd(),'datasets','tightdata.pickle')
        with open(obpath,'rb') as f:
            trainList, testList = pickle.load(f)
        if train:
            self.datalist = trainList
        else:
            self.datalist = testList
        self.mapw = {'T':1,'F':0}
        print('Lets see a datum:',self.datalist[15][0].shape)
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.datalist[idx][0]
        y = self.datalist[idx][1]
        
        # sample = {'mfcc':x,'label':y}
        return torch.from_numpy( np.array(x, dtype='float32') ), self.mapw[y]
    def separate(self):
        temp = list(zip(*self.datalist))
        x = list(temp[0])
        y = [ self.mapw[i] for i in temp[1] ]
        return x, y
    
    def packed_separate(self):
        temp = list(zip(*self.datalist))
        x = list(temp[0])
        x.sort(key=lambda z: len(z), reverse=True)
        x = [torch.from_numpy(i.astype(np.float32)) for i in x]
        lengths = [len(i) for i in x]
        # pdb.set_trace()
        withzero = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        p_x = rnn_utils.pack_padded_sequence(withzero, lengths, batch_first=True)
        y = [ self.mapw[i] for i in temp[1] ]
        return p_x, np.array(y)

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    
    xb, yb = zip(*data)
    # pdb.set_trace()
    xb = rnn_utils.pad_sequence(xb, batch_first=True, padding_value=0)
    # print(xb[0])
    # pdb.set_trace()
    # data = list(zip(xb,yb))
    return torch.from_numpy(np.array(xb)), torch.from_numpy(np.array(yb)), data_length
        
if __name__=='__main__':
    data = BSdataset()
    temp=list(zip(*data.datalist))
    pdb.set_trace()
    data_loader = DataLoader(data, batch_size=3, shuffle=True, 
                             collate_fn=collate_fn)
    batch_x, batch_y, batch_len = iter(data_loader).next()
    print(batch_x,'\n',batch_y)
    batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, 
                                                  batch_len, batch_first=True)
    # pdb.set_trace()
    for step, samples in enumerate(data_loader):
        b_x, b_y, b_len = samples[0],samples[1], samples[2]
        b_x_pack = rnn_utils.pack_padded_sequence(b_x, 
                                                  b_len, batch_first=True)
        pdb.set_trace()
    print('END')
    
    
    