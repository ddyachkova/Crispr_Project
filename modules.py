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




def calc_pres(y_true, y_pred): 
    y_true = y_true.cpu().detach().numpy()
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    y_pred_tags = y_pred_tags.cpu().detach().numpy()

    pres_score = precision_score(y_true, y_pred_tags, average='micro')
    f1 = f1_score(y_true, y_pred_tags, average='macro')

    return pres_score, f1

def do_eval(batch_size, num_iter, model, criterion, epoch):
    loss_ = []
    a = time()
    model.eval()
    monitor_step = 10
    for i in range(num_iter):
        X = get_batch(0 + i*batch_size, batch_size*(i+1), batch_size)
        y = y_train[0 + i*batch_size : batch_size*(i+1)]
        X, y =  torch.tensor(X), torch.tensor(y)        
        X, y = X.cuda(), y.cuda()
        
#         X = X.reshape(X.shape[1], X.shape[2])
#         y = y.reshape(y.shape[1])
        X = X.float()
        y = y.long()
        
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss_.append(loss.item())
        pres, f1 = calc_pres(y, y_pred)

        if i % monitor_step == 0: 
            print('Iteration {}: val loss: {} precision {}, f-1 {}'.format(i, loss.item(), pres, f1))

    loss_ = np.array(loss_)  
    s = '%d: Val loss:%f, MSE: N samples: %d in %f min'%(epoch, loss_.mean(), len(loss_), (time() -a)/60.)
    print(s)
    return loss_


def get_batch(start, end, batch_size, CSC_data):
    CSC_data_batch = CSC_data[(CSC_data.iloc[:, 0] >= start) & (CSC_data.iloc[:, 0] < end)]
    indices = CSC_data_batch.iloc[:, 0].values - start
    indptr = CSC_data_batch.iloc[:, 1].values
#     print (indices.max(), indices.min(), indptr.max())
    data = CSC_data_batch.iloc[:, 2].values
    mtx = sparse.csc_matrix((data, (indices, indptr)), shape=(batch_size, 149321)).toarray()
    return mtx


def train(y_train, CSC_data, batch_size, num_iter, model, optimizer, criterion, n_epochs, scheduler, res_name, debug=False):
    model.cuda()
    model.train()

    a = time()
    epoch = 0
    monitor_step = 10
    val_loss_, train_loss_, pres_ = [], [], []
    for e in range(n_epochs):
        epoch = e+1
        print('>>>>>>>Epoch %d'%(epoch))
        print(">>>>>>> Training")
        for i in range(num_iter):
            X_tr = get_batch(0 + i*batch_size, batch_size*(i+1), batch_size, CSC_data)
            y_tr = y_train[0 + i*batch_size : batch_size*(i+1)]
            X_tr, y_tr =  torch.tensor(X_tr), torch.tensor(y_tr)        
            X_tr, y_tr = X_tr.cuda(), y_tr.cuda()

#             X_tr = X_tr.reshape(X_tr.shape[1], X_tr.shape[2])
#             y_tr = y_tr.reshape(y_tr.shape[1])
            
            X_tr = X_tr.float()
            y_tr = y_tr.long()

            y_pred = model(X_tr)

            if debug:
                print ('X', X_tr.shape)
                print ('y_tr', y_tr.shape)
                print ('y_pred', y_pred.shape)

            loss = criterion(y_pred, y_tr)
            pres, f1 = calc_pres(y_tr, y_pred)


            train_loss_.append(loss.item())
            pres_.append(pres)

            if i % monitor_step == 0: 
                print('Epoch {} iteration {}: train loss: {} precision {} f1 {}'.format(epoch, i, loss.item(), pres, f1))
            loss.backward()
            optimizer.step()

        val_loss = do_eval(batch_size, num_iter, model, criterion, epoch)
        val_loss_.append(val_loss)
        if scheduler is not None: 
            scheduler.step()

    print('%d: Train time:%.2f min in %d steps'%(epoch, (time() - a)/60, len(train_loader)))
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

