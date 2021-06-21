import numpy as np 
import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import itertools
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from torch.optim.lr_scheduler import StepLR

from scipy.stats import spearmanr
from scipy import sparse

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, explained_variance_score, r2_score, roc_auc_score, precision_score, f1_score

# from modules import *
# from models import *

import matplotlib.pyplot as plt
import seaborn as sns

import random


class Dset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_path, y_path, dset_len, chunksize):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.chunksize = chunksize
        self.X_path = X_path
        self.y_path = y_path
        self.dset_len = dset_len
        self.reader_X = pd.read_csv(self.X_path) #, chunksize = self.chunksize) #, iterator=True)
        self.reader_y = pd.read_csv(self.y_path, chunksize = self.chunksize) #, iterator=True)

    def __len__(self):
        return self.dset_len

    def __getitem__(self, idx):
        self.X = self.get_batch(self.reader_X, self.chunksize, idx)
        self.y = self.reader_y.get_chunk(self.chunksize)
        self.y = self.y.iloc[:, 1:].replace(-1, 2).values
        
        return torch.tensor(self.X), torch.tensor(self.y)
    
    def get_batch(self, reader_X, chunksize, idx):
        X_batch = reader_X[(reader_X.iloc[:, 0] >= chunksize*idx) & (reader_X.iloc[:, 0] < chunksize*(idx + 1))]
        indices = X_batch.iloc[:, 0].values - chunksize*idx
        indptr = X_batch.iloc[:, 1].values
        data = X_batch.iloc[:, 2].values
        mtx = sparse.csc_matrix((data, (indices, indptr)), shape=(chunksize, 149321)).toarray()
        return mtx


def calc_pres(y_true, y_pred): 
    y_true = y_true.cpu().detach().numpy()
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    y_pred_tags = y_pred_tags.cpu().detach().numpy()

    pres_score = precision_score(y_true, y_pred_tags, average='micro')
    f1 = f1_score(y_true, y_pred_tags, average='macro')

    return pres_score, f1

def get_train_val_loader(X_path, y_path, len_dset, batch_size):
    custom_dset = Dset(X_path, y_path, len_dset, batch_size)
    dataset_indices = list(range(len(custom_dset)))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(0.2 * len(dataset_indices)))

    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=custom_dset, batch_size=1, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset=custom_dset, batch_size=1, shuffle=False, sampler=val_sampler)
    return train_loader, val_loader

def do_eval(val_loader, model, criterion, epoch):
    loss_ = []
    a = time()
    model.eval()
    monitor_step = 10
    for i, data in enumerate(val_loader):
        X, y = data
        X, y = X.cuda(), y.cuda()
        X = X.reshape(X.shape[1], X.shape[-1]).float()
        y = y.reshape(y.shape[1], y.shape[-1]).long()
        
        y_pred = model(X)
#         loss = criterion(y_pred, y)
        loss = 0
        for i in range(0, 4):
            loss__ = criterion(y_pred[:, i, :], y[:, i])
            loss += loss__
        loss_ = np.append(loss_, loss.item())
#         pres, f1 = calc_pres(y, y_pred)
        loss_ = np.array(loss_)  

        if i % monitor_step == 0: 
            print('Iteration {}: val loss: {} precision {}, f-1 {}'.format(i, loss.item(), pres, f1))
    s = '%d: Val loss:%f, MSE: N samples: %d in %f min'%(epoch, loss_.mean(), len(loss_), (time() -a)/60.)
    print(s)
    return loss_


# def get_batch(start, end, batch_size, CSC_data):
#     CSC_data_batch = CSC_data[(CSC_data.iloc[:, 0] >= start) & (CSC_data.iloc[:, 0] < end)]
#     indices = CSC_data_batch.iloc[:, 0].values - start
#     indptr = CSC_data_batch.iloc[:, 1].values
# #     print (indices.max(), indices.min(), indptr.max())
#     data = CSC_data_batch.iloc[:, 2].values
#     mtx = sparse.csc_matrix((data, (indices, indptr)), shape=(batch_size, 149321)).toarray()
#     return mtx


def train(X_path, y_path, len_dset, batch_size, model, optimizer, criterion, n_epochs, scheduler, res_name, debug=False):
    model.cuda()
    model.train()
    a = time()
    epoch = 0
    monitor_step = 2
    val_loss_, train_loss_, pres_ = [], [], []
    train_loader, val_loader = get_train_val_loader(X_path, y_path, len_dset, batch_size)
    for e in range(n_epochs):
        epoch = e+1
        print('>>>>>>>Epoch %d'%(epoch))
        print(">>>>>>> Training")
        for i, data in enumerate(train_loader):
            X_tr, y_tr = data
            X_tr = X_tr.reshape(X_tr.shape[1], X_tr.shape[-1]).float().cuda()
            y_tr = y_tr.reshape(y_tr.shape[1], y_tr.shape[-1]).long().cuda()


            y_pred = model(X_tr)

            if debug:
                print ('X', X_tr.shape)
                print ('y_tr', y_tr.shape)
                print ('y_pred', y_pred.shape)
            
            loss = 0
            for j in range(0, 4):
                loss_ = criterion(y_pred[:, j, :], y_tr[:, j])
                loss += loss_
                
#             pres, f1 = calc_pres(y_tr, y_pred)


            train_loss_.append(loss.item())
#             pres_.append(pres)
            if i % monitor_step == 0: 
#                 print('Epoch {} iteration {}: train loss: {} precision {} f1 {}'.format(epoch, i, loss.item(), pres, f1))
                print('Epoch {} iteration {}: train loss: {}'.format(epoch, i, loss.item()))

            loss.backward()
            optimizer.step()

        val_loss = do_eval(val_loader, model, criterion, epoch)
        val_loss_.append(val_loss)
        if scheduler is not None: 
            scheduler.step()

    print('%d: Train time:%.2f min in %d steps'%(epoch, (time() - a)/60, n_iter))
    model_save(model, optimizer, n_epochs, train_loss_, val_loss_, res_name)
    return train_loss_, val_loss_

def result_name(res_dir, num_epochs, model_class, dim1, dim2, dim3):
        try: 
            model_ind = [int(model_file.split('_')[1]) for model_file in os.listdir() if model_file.split('.')[-1] == 'pt'].max()
        except: 
            model_ind = 0
        res_str = 'model_{}_class_{}_{}_{}_{}_{}_epoch'.format(str(model_ind), model_class, str(dim1), str(dim2), str(dim3), num_epochs)
        res_name = os.path.join(res_dir, res_str) 
        res_name += '.pt'
        return res_name



def model_save(model, optimizer, epoch, training_loss, val_loss, res_name):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': training_loss,
            'val loss' : val_loss,
            }, 
        (res_name + '.pt'))

