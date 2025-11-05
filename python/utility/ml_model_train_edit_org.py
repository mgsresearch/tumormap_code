
""" edit files for the model training"""

import os 
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

# pytorch related 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml_deeplearning_edit_org import *

# define the random seed 
np.random.seed(42)

class EarlyStopping():

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        :param patience: How long (epochs) to wait after last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.filepath = "./data_mlp/mlp_trained_models/"

    def __call__(self, val_loss, model,epoch, loss,optimizer, file_name= "checkpoint.pt"):
            
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
            # torch.save({'epoch': epoch,
            #             'model_state_dict': model.state_dict(),
            #              'optimizer_state_dict': optimizer.state_dict(),
            #               'loss': loss }, self.filepath + file_name)
            # torch.save(model.state_dict(), self.filepath + file_name)
            print("current val loss is {}".format(self.val_loss_min))
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
class TumorMAP():

    def __init__(self, parse, path_of_dataset = [] ):

        # system path
        self.path_of_dataset = path_of_dataset

        # model training parameters
        self.gpu             = parse["gpu"]
        self.use_cpu         = parse["use_cpu"]
        self.batch_size      = parse["batch_size"]
        self.num_epochs      = parse["num_epochs"]
        self.lr              = parse["lr"]
        self.model_name      = parse["model_name"]
        self.log_dir         = parse["log_dir"]
        self.flag            = parse["flag"]
        self.tumor_type      = parse["tumor_type"]
        self.log_instance    = parse["log_instance"]
        self.path_save_model = parse["path_save_model"]
        self.loss_mode       = parse["loss_mode"]

    def dataset_normalize(self, dataset):
        """accepts entire dataset including the train and test set and finds mean and std of it before training
        input: dataset npy format data
        output-mean:
        output-std :
        """

        # considering the tumorid as a 1D signal        
        val_mean    = np.mean(dataset)
        val_std     = np.std(dataset)

        # val_mean    = np.mean(dataset, axis = 0)
        # val_std     = np.std(dataset, axis = 0)
        
        return val_mean, val_std
    
    def accuracy(self,outputs, labels):
        
        # obtain the prediction from the raw outputs
        _, preds = torch.max(outputs, 1)
        
        # predicted label
        # 1. if label(0) > label(1) -> use 0
        # 2. if label(1) > label(0) -> use 1
        # if we only have two classes -> how this is related to the comparison and the outputs? 
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
        
    def train_model(self,model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, early_stopper):
        
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        # checking  
        # print("self.loss_mode = ", self.loss_mode)
        # exit()

        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
         
            for batch_id, (data, target) in tqdm(enumerate(train_dataloader,0), total = len(train_dataloader), smoothing = 0.9):

                # reset the optimizer 
                optimizer.zero_grad()

                # produce the output from the forward round
                outputs = model(data)

                # loss function for binary classification 
                if self.loss_mode == "cross_entropy":
                    loss = criterion(outputs, target)
                    
                elif self.loss_mode == "NLLLoss":
                    outputs = torch.log(outputs)
                    loss = criterion(outputs, target)
                
                # update epoch accuracy
                epoch_acc += self.accuracy(outputs, target)

                # background propagation to update the gradients
                loss.backward()

                # update the model's parameters based on the computed gradients. 
                optimizer.step()
                
                # update the loss items
                epoch_loss += loss.item()
            
            """model validation or verification"""
            # fix to the evaluation mode 
            model.eval()
            val_loss = 0.0
            val_acc  = 0.0
            with torch.no_grad():
                for batch_id, (data, target) in tqdm(enumerate(val_dataloader,0), total = len(val_dataloader), smoothing = 0.9):

                    # generate the output from the data
                    outputs = model(data)
                    
                    # calculate the loss function
                    if self.loss_mode == "cross_entropy":
                        loss = criterion(outputs, target)
                    elif self.loss_mode == "NLLLoss":
                        outputs = torch.log(outputs)
                        loss = criterion(outputs, target)

                    # estimate the validation loss
                    val_loss += loss.item()
                    val_acc += self.accuracy(outputs, target)

                    # scheduler.step() is used to update the learning rate of the optimizer according to a predefined learning rate schedule.
                    scheduler.step()

            print("Epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".format(epoch+1, epoch_loss/len(train_dataloader), epoch_acc/len(train_dataloader), val_loss/len(val_dataloader), val_acc/len(val_dataloader)))
            self.writer.add_scalar("loss/train", epoch_loss/len(train_dataloader), epoch+1)
            self.writer.add_scalar("acc/train", epoch_acc/len(train_dataloader), epoch+1)
            self.writer.add_scalar("loss/val", val_loss/len(val_dataloader), epoch+1)
            self.writer.add_scalar("acc/val", val_acc/len(val_dataloader), epoch+1)

            train_loss_list.append(epoch_loss/len(train_dataloader))
            train_acc_list.append(epoch_acc/len(train_dataloader))
            val_loss_list.append(val_loss/len(val_dataloader))
            val_acc_list.append(val_acc/len(val_dataloader))

            if epoch > 50:
                if early_stopper(val_loss/len(val_dataloader), model, epoch, epoch_loss/len(train_dataloader),optimizer, file_name= self.model_name + "_" + self.log_instance + ".pt"):
                    if early_stopper.early_stop:
                        print("Early stopping at Epoch {}".format(epoch))
                        break
 
        self.writer.add_hparams(hparam_dict=self.hyperparams, metric_dict={"loss": epoch_loss/len(train_dataloader), "acc": epoch_acc/len(train_dataloader)}) # replace metric with test data 
        
        # self.plot_acc_loss(train_acc_list, train_loss_list, val_acc_list, val_loss_list)

        return train_acc_list, train_loss_list, val_acc_list, val_loss_list

    def build_model(self):
        """Build the MLP model"""
        
        # build the model
        model = SpectralClassification(loss_mode=self.loss_mode)
        # print(torch.cuda.is_available())
        # exit() 

        if self.use_cpu:
            # input("check")
            model = model.cpu()
        else:
            model = model.cuda(self.gpu)

        # define the loss function
        if self.loss_mode == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif self.loss_mode == "NLLLoss":
            criterion = nn.NLLLoss()
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        return model, criterion, optimizer
