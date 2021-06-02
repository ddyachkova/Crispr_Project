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

from modules import *
from models import *

import matplotlib.pyplot as plt
import seaborn as sns

import random
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 32, help='Batch size')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 10, type=int, help='Number of epochs')
    parser.add_argument('-num_iter', '--num_iter' , default = 2500, type=int, help='Number of iterations')
    
    
    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
    
    parser.add_argument('-model_class', '--model_class' , default = 'FeedforwardBin', type=str, help='Class of the model')
    parser.add_argument('-CSC_data_path', '--CSC_data_path',  type=str, help='Path to the input')
    parser.add_argument('-y_file_path', '--y_file_path', type=str, help='Path to the targets')
    parser.add_argument('-res_dir', '--res_dir', default = '.', type=str, help='Path to the targets')

    return parser.parse_args()


def main():
    args = get_args()
    print ('Got the arguments')
    CSC_data = pd.read_csv(args.CSC_data_path)
    y_train = pd.read_csv(args.y_file_path)
    print ('Got the data')
    CSC_data = CSC_data.iloc[:, 1:]
    y_train = y_train.iloc[:, 2].values
    
    class_weights = [4.8659e-05, 2.6752e-04] 
    dim1, dim2, dim3, dim4 = 149321, 512, 128, 2

    model = FeedforwardBin(dim1, dim2, dim3, dim4)
    print ('Initialized the model')

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.05)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weights).cuda())
    print ('Initialized the optimizer, scheduler, and criterion')

    res_name = result_name(args.res_dir, args.num_epochs, args.model_class, dim1, dim2, dim3)
    print ('Beginning the training')

    train_loss_, val_loss_= train(y_train, CSC_data, args.batch_size, args.num_iter, model, optimizer, criterion, args.num_epochs, scheduler, res_name, debug=False)
    
    
if __name__ == '__main__':
     main()
