
import os 
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import itertools
from itertools import permutations

# pytorch related 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# third-party
sys.path.append("./exp_mice_histopathology/mouse_mlp_study/")
sys.path.append("./utility/")

import ml_model_train_edit_org
import ml_inference_edit_org
 
random.seed(42)
np.random.seed(42)
torch.manual_seed(0)

class mlp_cross_validation_strategy(): 

    def __init__(self):
        """develop the system data loaders"""

        # TODO: define a single model 
        # single model for training 
        # single model for inference (prediction)

        # training mlp
        self._tumormap_agent   = [] 
    
        # global path for checking
        self._path_global_check       = []
        self._path_global_dataset     = []
        self._path_global_save_mlp    = []
        self._log_dir                 = []
        self._val_batchsize           = []
        self._val_numepoch            = []
        self._lr                      = []
        self._idx_use_gpu             = []
        self._is_use_cpu              = []
        self._val_tumor_type          = []
        self._val_log_instance        = []
        self._loss_mode               = [] 

    def mlp_train_with_os_two_to_one_combine(self):

        """os-tumor model 

        os_1273_tumor
        os_1274_tumor
        os_1276_tumor

        combinations = [ os_1273_tumor + os_1274_tumor + os_1276_tumor ]

        """

        # define the path of the results
        self.path_save_model        = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/pth_res/"
        self.path_save_acc          = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/acc_res/"
        self.path_save_loss         = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/loss_res/"
        self.path_dataset           = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/dataset/"
        self.model_name             = "tumormap"

        # define the path of the index
        list_of_group_idx           = [ ["os_1273_tumor", "os_1274_tumor"],
                                        ["os_1273_tumor", "os_1276_tumor"],
                                        ["os_1274_tumor", "os_1276_tumor"] ]
                    
        # define the path 
        list_of_group_idx_test_only = [ ["os_1276_tumor"],
                                        ["os_1274_tumor"], 
                                        ["os_1273_tumor"] ]
        
        # save to the .txt file (training + validation)
        with open( self.path_save_model + "group_idx_train_and_val.txt", 'w' ) as output:
            for item_group in list_of_group_idx:
                output.write(str( item_group ) + '\n')

        # save to the .txt file (testing only)
        with open( self.path_save_model + "group_idx_test.txt", 'w' ) as output:
            for item_group in list_of_group_idx_test_only:
                output.write(str( item_group ) + '\n')
   
        for idx_group, list_group_tmp in enumerate( list_of_group_idx ): 

            if idx_group > 0:
                break

            # define the group index
            self.log_instance           = "group_idx_" + str( idx_group )            

            # load the dataset
            data_arr_train      = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_data_train_and_val.npy" )
            data_label_train    = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_label_train_and_val.npy" )

            # mean and std of dataset
            # normalize dataset using mean and std (z score normalization)
            database_mean, database_std = self._tumormap_agent.dataset_normalize(data_arr_train)
            data_arr_train              = (data_arr_train - database_mean) / database_std

            # train and validation set
            x_train, x_val, y_train, y_val = train_test_split(data_arr_train, 
                                                              data_label_train, 
                                                              test_size     = 0.3, 
                                                              random_state  = 42, 
                                                              stratify      = data_label_train)

            # build the model architecture
            model, criterion, optimizer = self._tumormap_agent.build_model()
            scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            early_stopper               = ml_model_train_edit_org.EarlyStopping(patience=5, delta = 0.5)
        
            # convert the data to tensor
            x_train_tensor  = torch.from_numpy(x_train).float()
            y_train_tensor  = torch.from_numpy(y_train).long()
            x_val_tensor   = torch.from_numpy(x_val).float()
            y_val_tensor   = torch.from_numpy(y_val).long()

            if self._tumormap_agent.use_cpu:
                x_train_tensor  = x_train_tensor.cpu()
                y_train_tensor  = y_train_tensor.cpu()
                x_val_tensor    = x_val_tensor.cpu()
                y_val_tensor    = y_val_tensor.cpu()
            else:
                # load data to gpu
                x_train_tensor  = x_train_tensor.cuda( self._tumormap_agent.gpu )
                y_train_tensor  = y_train_tensor.cuda( self._tumormap_agent.gpu )
                x_val_tensor    = x_val_tensor.cuda( self._tumormap_agent.gpu )
                y_val_tensor    = y_val_tensor.cuda( self._tumormap_agent.gpu )

            dataset_train       = TensorDataset(x_train_tensor, y_train_tensor)
            dataset_val         = TensorDataset(x_val_tensor, y_val_tensor)

            self._tumormap_agent.hyperparams = {
                                            'lr' :          self._tumormap_agent.lr,
                                            'batch_size' :  self._tumormap_agent.batch_size,
                                            'epochs' :      self._tumormap_agent.num_epochs,
                                            'model_name' :  self._tumormap_agent.model_name,
                                            'tumor_type' :  self._tumormap_agent.tumor_type,
                                            "training_set": len(dataset_train),
                                            "test_set":     len(dataset_val)
                                                }

            train_dataloader = DataLoader(dataset_train, batch_size = self._tumormap_agent.batch_size, shuffle=True)     
            val_dataloader = DataLoader(dataset_val, batch_size = self._tumormap_agent.batch_size, shuffle=True) 

            # build the summary workflog
            self._tumormap_agent.writer = SummaryWriter( self._tumormap_agent.log_dir + "/" + self.log_instance )

            # train the model
            train_acc_list, train_loss_list, val_acc_list, val_loss_list = self._tumormap_agent.train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,early_stopper)

            # save the model
            # torch.save(model.state_dict(), self.path_save_model + self.model_name + "_" + self.log_instance + ".pth")
            # print("Model saved successfully")

            # # save the acc and loss
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_train_acc.npy", train_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_train_loss.npy", train_loss_list )
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_val_acc.npy", val_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_val_loss.npy", val_loss_list )

            self._tumormap_agent.writer.close()

    def mlp_train_with_four_to_one_combine(self):
        """sts-tumor model
        
            with the four-to-one combinations
            sts_1288_tumor
            sts_1289_tumor
            sts_1293_tumor
            sts_1294_tumor
            sts_1266_tumor

        combinations = [ sts_1288_tumor + sts_1289_tumor + sts_1293_tumor + sts_1294_tumor ]

        """

        # define the path of the results
        self.path_save_model        = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/pth_res/"
        self.path_save_acc          = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/acc_res/"
        self.path_save_loss         = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/loss_res/"
        self.path_dataset           = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/dataset/"
        self.model_name             = "tumormap"

        # define the path of the index
        list_of_group_idx           = [ ["sts_1288_tumor", "sts_1289_tumor", "sts_1293_tumor", "sts_1294_tumor"],
                                        ["sts_1288_tumor", "sts_1289_tumor", "sts_1293_tumor", "sts_1266_tumor"], 
                                        ["sts_1288_tumor", "sts_1289_tumor", "sts_1294_tumor", "sts_1266_tumor"], 
                                        ["sts_1288_tumor", "sts_1293_tumor", "sts_1294_tumor", "sts_1266_tumor"], 
                                        ["sts_1289_tumor", "sts_1293_tumor", "sts_1294_tumor", "sts_1266_tumor"]  ]
                    
        # define the path 
        list_of_group_idx_test_only = [ ["sts_1266_tumor"],
                                        ["sts_1294_tumor"], 
                                        ["sts_1293_tumor"], 
                                        ["sts_1289_tumor"], 
                                        ["sts_1288_tumor"] ]
        
        # training + validation -> txt file 
        with open( self.path_save_model + "group_idx_train_and_val.txt", 'w' ) as output:
            for item_group in list_of_group_idx:
                output.write(str( item_group ) + '\n')
        # testing -> txt
        with open( self.path_save_model + "group_idx_test.txt", 'w' ) as output:
            for item_group in list_of_group_idx_test_only:
                output.write(str( item_group ) + '\n')
   
        for idx_group, list_group_tmp in enumerate( list_of_group_idx ): 

            if idx_group > 0:
                break 

            # define the group index
            self.log_instance           = "group_idx_" + str( idx_group )            

            # load the dataset
            data_arr_train   = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_data_train_and_val.npy" )
            data_label_train = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_label_train_and_val.npy" )

            # mean and std of dataset
            # normalize dataset using mean and std (z score normalization)
            database_mean, database_std = self._tumormap_agent.dataset_normalize(data_arr_train)
            data_arr_train              = (data_arr_train - database_mean) / database_std

            # train and validation
            x_train, x_val, y_train, y_val = train_test_split(data_arr_train, 
                                                              data_label_train, 
                                                              test_size     = 0.3, 
                                                              random_state  = 42, 
                                                              stratify      = data_label_train)

            # build the model architecture
            model, criterion, optimizer = self._tumormap_agent.build_model()
            scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            early_stopper               = ml_model_train_edit_org.EarlyStopping(patience=5, delta = 0.5)
        
            # convert the data to tensor
            x_train_tensor  = torch.from_numpy(x_train).float()
            y_train_tensor  = torch.from_numpy(y_train).long()
            x_val_tensor   = torch.from_numpy(x_val).float()
            y_val_tensor   = torch.from_numpy(y_val).long()

            if self._tumormap_agent.use_cpu:
                x_train_tensor  = x_train_tensor.cpu()
                y_train_tensor  = y_train_tensor.cpu()
                x_val_tensor    = x_val_tensor.cpu()
                y_val_tensor    = y_val_tensor.cpu()
            else:
                # load data to gpu
                x_train_tensor  = x_train_tensor.cuda( self._tumormap_agent.gpu )
                y_train_tensor  = y_train_tensor.cuda( self._tumormap_agent.gpu )
                x_val_tensor    = x_val_tensor.cuda( self._tumormap_agent.gpu )
                y_val_tensor    = y_val_tensor.cuda( self._tumormap_agent.gpu )

            dataset_train       = TensorDataset(x_train_tensor, y_train_tensor)
            dataset_val         = TensorDataset(x_val_tensor, y_val_tensor)

            self._tumormap_agent.hyperparams = {
                                            'lr' :          self._tumormap_agent.lr,
                                            'batch_size' :  self._tumormap_agent.batch_size,
                                            'epochs' :      self._tumormap_agent.num_epochs,
                                            'model_name' :  self._tumormap_agent.model_name,
                                            'tumor_type' :  self._tumormap_agent.tumor_type,
                                            "training_set": len(dataset_train),
                                            "test_set":     len(dataset_val)
                                                }

            train_dataloader = DataLoader(dataset_train, batch_size = self._tumormap_agent.batch_size, shuffle=True)     
            val_dataloader = DataLoader(dataset_val, batch_size = self._tumormap_agent.batch_size, shuffle=True) 
            # val_dataloader = DataLoader(dataset_test, batch_size = 5, shuffle=True) 

            # build the summary workflog
            self._tumormap_agent.writer = SummaryWriter( self._tumormap_agent.log_dir + "/" + self.log_instance )

            # train the model
            train_acc_list, train_loss_list, val_acc_list, val_loss_list = self._tumormap_agent.train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,early_stopper)

            # save the model
            # torch.save(model.state_dict(), self.path_save_model + self.model_name + "_" + self.log_instance + ".pth")
            # print("Model saved successfully")

            # save the acc and loss
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_train_acc.npy", train_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_train_loss.npy", train_loss_list )
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_val_acc.npy", val_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_val_loss.npy", val_loss_list )

            # self._tumormap_agent.writer.close()

    def mlp_train_with_three_to_two_combine(self):

        """MLP model training
            STS-tumors
            "sts_1288_tumor" = 0 
            "sts_1289_tumor" = 1
            "sts_1293_tumor" = 2
            "sts_1294_tumor" = 3
            "sts_1266_tumor" = 4 
            10 combinations
        """ 

        # result_mlp_cross_validation_three_to_two_sts
        self.path_save_model        = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/pth_res/"
        self.path_save_acc          = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/acc_res/"
        self.path_save_loss         = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/loss_res/"
        self.path_dataset           = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/dataset/"
        self.model_name             = "tumormap"

        # C_3_2 = 5 * 4 * 3 / 6 = 60 / 6 = 10
        list_of_group_idx           = [ ["sts_1288_tumor", "sts_1289_tumor", "sts_1293_tumor"],
                                        ["sts_1288_tumor", "sts_1289_tumor", "sts_1294_tumor"], 
                                        ["sts_1288_tumor", "sts_1289_tumor", "sts_1266_tumor"], 
                                        ["sts_1288_tumor", "sts_1293_tumor", "sts_1294_tumor"], 
                                        ["sts_1288_tumor", "sts_1293_tumor", "sts_1266_tumor"], 
                                        ["sts_1288_tumor", "sts_1294_tumor", "sts_1266_tumor"], 
                                        ["sts_1289_tumor", "sts_1293_tumor", "sts_1294_tumor"], 
                                        ["sts_1289_tumor", "sts_1293_tumor", "sts_1266_tumor"], 
                                        ["sts_1289_tumor", "sts_1294_tumor", "sts_1266_tumor"], 
                                        ["sts_1293_tumor", "sts_1294_tumor", "sts_1266_tumor"]  ]
        
        list_of_group_idx_test_only = [ ["sts_1294_tumor", "sts_1266_tumor"],
                                        ["sts_1293_tumor", "sts_1266_tumor"], 
                                        ["sts_1293_tumor", "sts_1294_tumor"], 
                                        ["sts_1289_tumor", "sts_1266_tumor"], 
                                        ["sts_1289_tumor", "sts_1294_tumor"], 
                                        ["sts_1289_tumor", "sts_1293_tumor"], 
                                        ["sts_1288_tumor", "sts_1266_tumor"], 
                                        ["sts_1288_tumor", "sts_1294_tumor"], 
                                        ["sts_1288_tumor", "sts_1293_tumor"], 
                                        ["sts_1288_tumor", "sts_1289_tumor"]  ]

        # training + validation -> .txt file 
        # save to the .txt file 
        with open( self.path_save_model + "group_idx_train_and_val.txt", 'w' ) as output:
            # print("self.path_save_model = ", self.path_save_model) 
            for item_group in list_of_group_idx:
                output.write(str( item_group ) + '\n')
        # testing -> .txt file 
        with open( self.path_save_model + "group_idx_test.txt", 'w' ) as output:
            # print("self.path_save_model = ", self.path_save_model) 
            for item_group in list_of_group_idx_test_only:
                output.write(str( item_group ) + '\n')
   
        for idx_group, list_group_tmp in enumerate( list_of_group_idx ): 

            if idx_group > 0:
                break 

            # list_group_tmp == training one 
            self.log_instance           = "group_idx_" + str( idx_group )            

            # load the dataset 
            data_arr_train   = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_data_train_and_val.npy" )
            data_label_train = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_label_train_and_val.npy" )

            # # # checking  
            # data_check_1 = data_arr_train[0,:]
            # print("data_arr_train shape = ", data_arr_train.shape)
            # print("data_check_1 shape = ", data_check_1.shape)
            # exit()

            # mean and std of dataset
            # normalize dataset using mean and std (z score normalization)
            database_mean, database_std = self._tumormap_agent.dataset_normalize(data_arr_train)
            data_arr_train              = (data_arr_train - database_mean) / database_std

            # train and validation set
            x_train, x_val, y_train, y_val = train_test_split(data_arr_train, 
                                                              data_label_train, 
                                                              test_size = 0.3, 
                                                              random_state = 42, 
                                                              stratify = data_label_train)

            # build the model architecture
            model, criterion, optimizer = self._tumormap_agent.build_model()
            scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            early_stopper               = ml_model_train_edit_org.EarlyStopping(patience=5, delta = 0.5)
        
            # checking 
            # print("x_train shape = ", x_train.shape)
            # exit()

            # convert the data to tensor
            x_train_tensor      = torch.from_numpy(x_train).float()
            y_train_tensor      = torch.from_numpy(y_train).long()
            x_val_tensor        = torch.from_numpy(x_val).float()
            y_val_tensor        = torch.from_numpy(y_val).long()

            # checking 
            # print("x_train_tensor shape = ", x_train_tensor.shape)
            # exit()

            if self._tumormap_agent.use_cpu:
                x_train_tensor  = x_train_tensor.cpu()
                y_train_tensor  = y_train_tensor.cpu()
                x_val_tensor    = x_val_tensor.cpu()
                y_val_tensor    = y_val_tensor.cpu()
            else:
                # load data to gpu
                x_train_tensor  = x_train_tensor.cuda( self._tumormap_agent.gpu )
                y_train_tensor  = y_train_tensor.cuda( self._tumormap_agent.gpu )
                x_val_tensor    = x_val_tensor.cuda( self._tumormap_agent.gpu )
                y_val_tensor    = y_val_tensor.cuda( self._tumormap_agent.gpu )

            dataset_train       = TensorDataset(x_train_tensor, y_train_tensor)
            dataset_val         = TensorDataset(x_val_tensor, y_val_tensor)

            self._tumormap_agent.hyperparams = {
                                            'lr' :          self._tumormap_agent.lr,
                                            'batch_size' :  self._tumormap_agent.batch_size,
                                            'epochs' :      self._tumormap_agent.num_epochs,
                                            'model_name' :  self._tumormap_agent.model_name,
                                            'tumor_type' :  self._tumormap_agent.tumor_type,
                                            "training_set": len(dataset_train),
                                            "test_set":     len(dataset_val)
                                                }

            train_dataloader    = DataLoader(dataset_train, batch_size = self._tumormap_agent.batch_size, shuffle=True)     
            val_dataloader      = DataLoader(dataset_val, batch_size = self._tumormap_agent.batch_size, shuffle=True) 

            # build the summary workflog
            self._tumormap_agent.writer = SummaryWriter( self._tumormap_agent.log_dir + "/" + self.log_instance )

            # train the model
            train_acc_list, train_loss_list, val_acc_list, val_loss_list = self._tumormap_agent.train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,early_stopper)

            # save the model
            # torch.save(model.state_dict(), self.path_save_model + self.model_name + "_" + self.log_instance + ".pth")
            # print("Model saved successfully")

            # save the acc and loss
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_train_acc.npy", train_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_train_loss.npy", train_loss_list )
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_val_acc.npy", val_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_val_loss.npy", val_loss_list )

            self._tumormap_agent.writer.close()

class test_internal(): 

    def __init__(self):

        # define the inference class
        self._inference_class_official = mlp_cross_validation_strategy() 

        # inference object
        self._unit_mlp_infer_class = ml_inference_edit_org.mlp_inference_unit_test()

    def init_mlp_os_two_to_one_setting(self): 
        """os-model for the two-to-one combination"""

        # global path for checking
        self._inference_class_official._path_global_check       = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/data_loss_acc_result/"
        self._inference_class_official._path_global_dataset     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/data_tumormap_official/"
        self._inference_class_official._path_global_save_mlp    = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/data_mlp_model_official/"
        self._inference_class_official._log_dir                 = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/mlp_train_log_dir/"

        self._inference_class_official._val_batchsize           = 16
        self._inference_class_official._val_numepoch            = 150
        self._inference_class_official._lr                      = 0.001
        self._inference_class_official._idx_use_gpu             = 0
        self._inference_class_official._is_use_cpu              = False

        # this is updated
        self._inference_class_official._val_tumor_type          = "OS"
        
        self._inference_class_official._val_log_instance        = "sts_histo_log"
        self._inference_class_official._loss_mode               = "NLLLoss"

    def init_mlp_sts_four_to_one_setting(self):  
        """sts-tumor model for the four-to-one combination"""

        # global path for checking
        # idx = result_mlp_cross_validation_four_to_one_sts 
        self._inference_class_official._path_global_check       = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/data_loss_acc_result/"
        self._inference_class_official._path_global_dataset     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/data_tumormap_official/"
        self._inference_class_official._path_global_save_mlp    = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/data_mlp_model_official/"
        self._inference_class_official._log_dir                 = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/mlp_train_log_dir/"

        self._inference_class_official._val_batchsize           = 16
        self._inference_class_official._val_numepoch            = 150
        self._inference_class_official._lr                      = 0.001
        self._inference_class_official._idx_use_gpu             = 0
        self._inference_class_official._is_use_cpu              = False
        self._inference_class_official._val_tumor_type          = "STS"
        self._inference_class_official._val_log_instance        = "sts_histo_log"
        self._inference_class_official._loss_mode               = "NLLLoss"

    def init_mlp_sts_three_to_two_setting(self): 
        """ global paths + parameter settings
            1.  data_loss_acc_result
            2.  data_tumormap_official
            3.  data_mlp_model_official
            4.  mlp_train_log_dir
        """

        # global path for checking
        self._inference_class_official._path_global_check       = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/data_loss_acc_result/"
        self._inference_class_official._path_global_dataset     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/data_tumormap_official/"
        self._inference_class_official._path_global_save_mlp    = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/data_mlp_model_official/"
        self._inference_class_official._log_dir                 = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/mlp_train_log_dir/"

        self._inference_class_official._val_batchsize           = 16
        self._inference_class_official._val_numepoch            = 150
        self._inference_class_official._lr                      = 0.001
        self._inference_class_official._idx_use_gpu             = 0
        self._inference_class_official._is_use_cpu              = False
        self._inference_class_official._val_tumor_type          = "STS"
        self._inference_class_official._val_log_instance        = "sts_histo_log"
        self._inference_class_official._loss_mode               = "NLLLoss"

    def init_mlp_train_model(self): 

        # parse the input arguments
        parse_train_mlp = {
                            "gpu":              self._inference_class_official._idx_use_gpu,
                            "use_cpu":          self._inference_class_official._is_use_cpu,               # True
                            "batch_size":       self._inference_class_official._val_batchsize,            # 16,
                            "num_epochs":       self._inference_class_official._val_numepoch,             # 150,
                            "lr":               self._inference_class_official._lr,
                            "model_name":       "tumor_map",
                            "path_save_model":  self._inference_class_official._path_global_save_mlp, 
                            "log_dir":          self._inference_class_official._log_dir,
                            "flag":             "train",
                            "tumor_type":       self._inference_class_official._val_tumor_type,                # "STS" or "OS"
                            "log_instance":     self._inference_class_official._val_log_instance,              # "sts_official_1" or "os_official_1"
                            "loss_mode":        self._inference_class_official._loss_mode                      # NLLLoss or cross_entropy
                           }
          
        # this aims to deliver the variable4 to above class ( this is im )
        self._inference_class_official._tumormap_agent = ml_model_train_edit_org.TumorMAP( parse = parse_train_mlp, 
                                                                                           path_of_dataset = self._inference_class_official._path_global_dataset ) 
        
        
        # checking the pre-defined settings
        print( "self._inference_class_official._path_global_dataset = ", self._inference_class_official._path_global_dataset )
        print( "parse_train_mlp = ", parse_train_mlp )
        print("self._mlp_tumormap_agent = ", self._inference_class_official._tumormap_agent)

    def mlp_train_cross_validation_two_to_one(self): 

        # idx = mlp_train_with_os_two_to_one_combine
        self._inference_class_official.mlp_train_with_os_two_to_one_combine() 

    def mlp_train_cross_validation_four_to_one(self): 

        # idx = mlp_train_with_four_to_one_combine
        self._inference_class_official.mlp_train_with_four_to_one_combine()

    def mlp_train_cross_validation_three_to_two(self): 
        """run the full MLP training"""

        # idx = mlp_train_with_three_to_two_combine
        self._inference_class_official.mlp_train_with_three_to_two_combine()

    def init_mlp_inference_model_os_two_to_one(self, idx_group = []): 
        """perform the inference model"""

        # unit-1: test the inference class 
        para_inference_input                   = {}
        para_inference_input["model_name"]     = "tumormap"
        para_inference_input["loss_mode"]      = "NLLLoss"
        para_inference_input["model_dict"]     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/pth_res/"
        para_inference_input["log_instance"]   = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]   = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )
        
        # unit-2-a: train mlp 
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # mean and std from the training dataset
        self._unit_mlp_infer_class._path_global_dataset = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/dataset/"
        path_data_use_for_training             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                 = np.load( path_data_use_for_training )
        mean_data_tmp, std_data_tmp            = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                    = mean_data_tmp
        self._std_data_tmp                     = std_data_tmp

        # print("path_data_use_for_training = ", path_data_use_for_training)
        # print("mean_data_tmp = ", mean_data_tmp)
        # print("std_data_tmp  = ", std_data_tmp)
        # input("check mean and std")
        num_of_train_and_val = data_npy_train_and_val.shape[0]

        return num_of_train_and_val

    def init_mlp_inference_model_four_to_one(self, idx_group = []): 
        """perform the inference model"""

        # unit-1: test the inference class 
        para_inference_input                   = {}
        para_inference_input["model_name"]     = "tumormap"
        para_inference_input["loss_mode"]      = "NLLLoss"
        para_inference_input["model_dict"]     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/pth_res/"
        para_inference_input["log_instance"]   = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]   = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )
        
        # unit-2-a: train mlp 
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # mean and std from the training dataset
        self._unit_mlp_infer_class._path_global_dataset = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/dataset/"
        path_data_use_for_training             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                 = np.load( path_data_use_for_training )
        mean_data_tmp, std_data_tmp            = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                    = mean_data_tmp
        self._std_data_tmp                     = std_data_tmp

        # print("path_data_use_for_training = ", path_data_use_for_training)
        # print("mean_data_tmp = ", mean_data_tmp)
        # print("std_data_tmp  = ", std_data_tmp)
        # input("check mean and std")

        num_of_train_and_val = data_npy_train_and_val.shape[0]

        return num_of_train_and_val

    def init_mlp_inference_model_sts_three_to_two(self, idx_group = []): 
        """perform the inference model"""

        # unit-1: test the inference class 
        para_inference_input                   = {}
        para_inference_input["model_name"]     = "tumormap"
        para_inference_input["loss_mode"]      = "NLLLoss"
        para_inference_input["model_dict"]     = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/pth_res/"
        para_inference_input["log_instance"]   = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]   = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )
        
        # unit-2-a: train mlp 
        # this is to calculate the "dataset_normalize"
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # mean and std from the training dataset
        self._unit_mlp_infer_class._path_global_dataset = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/dataset/"
        path_data_use_for_training             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" + para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                 = np.load( path_data_use_for_training )
        mean_data_tmp, std_data_tmp            = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                    = mean_data_tmp
        self._std_data_tmp                     = std_data_tmp

        # print("path_data_use_for_training = ", path_data_use_for_training)
        # print("mean_data_tmp = ", mean_data_tmp)
        # print("std_data_tmp  = ", std_data_tmp)
        # input("check mean and std")

        num_of_train_and_val = data_npy_train_and_val.shape[0]

        return num_of_train_and_val

    def mlp_inference_cross_validation_os_two_to_one(self, idx_group_tmp = []):
        """inference model for the mice study"""

        if idx_group_tmp == 0:  
            list_group_test_mice = [ "os_1276_tumor_line_1", "os_1276_tumor_line_1" ]
        
        if idx_group_tmp == 1:  
            list_group_test_mice = [ "os_1274_tumor_line_1" ]

        if idx_group_tmp == 2:  
            list_group_test_mice = [ "os_1273_tumor_line_1" ]

        # statistics 
        list_of_infer        = []
        list_of_label        = []
        list_of_acc          = []
        list_of_tpr          = []
        list_of_tnr          = [] 
        list_of_fpr          = []
        list_of_fnr          = [] 
        list_of_cnr          = []

        count_acc_tmp        = 0
        count_tp_tmp         = 0 
        count_tn_tmp         = 0
        count_fp_tmp         = 0 
        count_fn_tmp         = 0 
        count_all_tmp        = 0 

        # summarize the prediction and the grond truth
        list_of_y_true_all      = [] 
        list_of_y_predict_all   = []

        for name_of_mice_tmp in list_group_test_mice:

            idx_scan_high_or_label              = "high_scan"

            # label 
            label_line_tmp = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_label.npy" )

            # inference 
            para_input_tmp                          = {} 
            para_input_tmp["mean_spectra"]          = self._mean_data_tmp
            para_input_tmp["std_spectra"]           = self._std_data_tmp
            para_input_tmp["loss_mode"]             = "NLLLoss"
            para_input_tmp["data_npy_stich_wv_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_wv.npy" )
            para_input_tmp["data_npy_stich_sp_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_sp.npy" )
            infer_line_tmp                          = self._unit_mlp_infer_class.infer_npy_data_check( para_input = para_input_tmp )

            # summary 
            res_infer_tmp = np.asanyarray( infer_line_tmp ).astype(np.uint8).tolist()
            res_label_tmp = label_line_tmp.tolist()

            list_of_infer = list_of_infer + res_infer_tmp
            list_of_label = list_of_label + res_label_tmp

            # statistics
            # TODO: implement the functions
            res_acc_sum, count_tp, count_tn, count_fp, count_fn, count_all, list_of_y_true, list_of_y_predict = statistics_from_inference(res_inference=res_infer_tmp, res_label=res_label_tmp)
            count_acc_tmp += res_acc_sum
            count_tp_tmp  += count_tp
            count_tn_tmp  += count_tn
            count_fp_tmp  += count_fp
            count_fn_tmp  += count_fn
            count_all_tmp += count_all

            list_of_y_true_all += list_of_y_true
            list_of_y_predict_all += list_of_y_predict

        accr = np.round( count_acc_tmp / count_all_tmp, 2 )
        tpr  = np.round( count_tp_tmp / ( count_tp_tmp + count_fn_tmp ), 2 )
        tnr  = np.round( count_tn_tmp / ( count_tn_tmp + count_fp_tmp ), 2 ) 
        fpr  = np.round( count_fp_tmp / ( count_fp_tmp + count_tn_tmp ), 2 )
        fnr  = np.round( count_fn_tmp / ( count_tp_tmp + count_fn_tmp ), 2 ) 

        accr = count_acc_tmp
        tpr  = count_tp_tmp
        tnr  = count_tn_tmp
        fpr  = count_fp_tmp
        fnr  = count_fn_tmp

        list_of_acc.append( accr ) 
        list_of_tpr.append(  tpr )
        list_of_tnr.append(  tnr  )
        list_of_fpr.append(  fpr  )
        list_of_fnr.append(  fnr )
        list_of_cnr.append(  count_all_tmp )

        print("list_of_infer = ", list_of_infer)
        print("list_of_label = ", list_of_label)
        print("list_of_tp = ", list_of_tpr)
        print("list_of_tn = ", list_of_tnr)
        print("list_of_fp = ", list_of_fpr)
        print("list_of_fn = ", list_of_fnr)

        return list_of_acc, list_of_tpr, list_of_tnr, list_of_fpr, list_of_fnr, list_of_cnr, list_of_y_true_all, list_of_y_predict_all

    def mlp_inference_cross_validation_four_to_one(self, idx_group_tmp = [] ):  
        """inference model for the mice study"""

        if idx_group_tmp == 0:  
            list_group_test_mice = [ "sts_1266_tumor_line_2" ]
        
        if idx_group_tmp == 1:  
            list_group_test_mice = [ "sts_1294_tumor_line_1", "sts_1294_tumor_line_2" ]

        if idx_group_tmp == 2:  
            list_group_test_mice = [ "sts_1293_tumor_line_1", "sts_1293_tumor_line_2" ]

        if idx_group_tmp == 3:  
            list_group_test_mice = [ "sts_1289_tumor_line_1", "sts_1289_tumor_line_2" ]

        if idx_group_tmp == 4:  
            list_group_test_mice = [ "sts_1288_tumor_line_1", "sts_1288_tumor_line_2" ]

        # statistics 
        list_of_infer        = []
        list_of_label        = []
        list_of_acc          = []
        list_of_tpr          = []
        list_of_tnr          = [] 
        list_of_fpr          = []
        list_of_fnr          = [] 
        list_of_cnr          = []
        
        # summarize the prediction and the grond truth
        list_of_y_true_all      = [] 
        list_of_y_predict_all   = []

        count_acc_tmp        = 0
        count_tp_tmp         = 0 
        count_tn_tmp         = 0
        count_fp_tmp         = 0 
        count_fn_tmp         = 0 
        count_all_tmp        = 0 

        for name_of_mice_tmp in list_group_test_mice:

            idx_scan_high_or_label              = "high_scan"

            # label
            label_line_tmp = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_label.npy" )

            # inference 
            para_input_tmp                          = {} 
            para_input_tmp["mean_spectra"]          = self._mean_data_tmp
            para_input_tmp["std_spectra"]           = self._std_data_tmp
            para_input_tmp["loss_mode"]             = "NLLLoss"
            para_input_tmp["data_npy_stich_wv_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_wv.npy" )
            para_input_tmp["data_npy_stich_sp_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_sp.npy" )
            infer_line_tmp                          = self._unit_mlp_infer_class.infer_npy_data_check( para_input = para_input_tmp )

            # summary 
            res_infer_tmp = np.asanyarray( infer_line_tmp ).astype(np.uint8).tolist()
            res_label_tmp = label_line_tmp.tolist()

            list_of_infer = list_of_infer + res_infer_tmp
            list_of_label = list_of_label + res_label_tmp

            # statistics
            # TODO: implement the functions
            res_acc_sum, count_tp, count_tn, count_fp, count_fn, count_all, list_of_y_true, list_of_y_predict = statistics_from_inference(res_inference=res_infer_tmp, res_label=res_label_tmp)
            count_acc_tmp += res_acc_sum
            count_tp_tmp  += count_tp
            count_tn_tmp  += count_tn
            count_fp_tmp  += count_fp
            count_fn_tmp  += count_fn
            count_all_tmp += count_all

            list_of_y_true_all += list_of_y_true
            list_of_y_predict_all += list_of_y_predict

        accr = np.round( count_acc_tmp / count_all_tmp, 2 )
        tpr  = np.round( count_tp_tmp / ( count_tp_tmp + count_fn_tmp ), 2 )
        tnr  = np.round( count_tn_tmp / ( count_tn_tmp + count_fp_tmp ), 2 ) 
        fpr  = np.round( count_fp_tmp / ( count_fp_tmp + count_tn_tmp ), 2 )
        fnr  = np.round( count_fn_tmp / ( count_tp_tmp + count_fn_tmp ), 2 ) 

        accr = count_acc_tmp
        tpr  = count_tp_tmp
        tnr  = count_tn_tmp
        fpr  = count_fp_tmp
        fnr  = count_fn_tmp

        list_of_acc.append( accr ) 
        list_of_tpr.append(  tpr )
        list_of_tnr.append(  tnr  )
        list_of_fpr.append(  fpr  )
        list_of_fnr.append(  fnr )
        list_of_cnr.append(  count_all_tmp )

        print("list_of_infer = ", list_of_infer)
        print("list_of_label = ", list_of_label)
        print("list_of_tp = ", list_of_tpr)
        print("list_of_tn = ", list_of_tnr)
        print("list_of_fp = ", list_of_fpr)
        print("list_of_fn = ", list_of_fnr)

        return list_of_acc, list_of_tpr, list_of_tnr, list_of_fpr, list_of_fnr, list_of_cnr, list_of_y_true_all, list_of_y_predict_all

    def mlp_inference_cross_validation_sts_three_to_two(self, idx_group_tmp = [] ):  
        """inference model for the testing-only mice data"""

        # TODO: implement the full functions of the systems 
        
        if idx_group_tmp == 0:  
            list_group_test_mice = [ "sts_1294_tumor_line_1", "sts_1294_tumor_line_2",  "sts_1266_tumor_line_2" ]
        
        if idx_group_tmp == 1:  
            list_group_test_mice = [ "sts_1293_tumor_line_1", "sts_1293_tumor_line_2",  "sts_1266_tumor_line_2" ]

        if idx_group_tmp == 2:  
            list_group_test_mice = [ "sts_1293_tumor_line_1", "sts_1293_tumor_line_2", "sts_1294_tumor_line_1", "sts_1294_tumor_line_2" ]

        if idx_group_tmp == 3:  
            list_group_test_mice = [ "sts_1289_tumor_line_1", "sts_1289_tumor_line_2", "sts_1266_tumor_line_2" ]

        if idx_group_tmp == 4:  
            list_group_test_mice = [ "sts_1289_tumor_line_1", "sts_1289_tumor_line_2", "sts_1294_tumor_line_1", "sts_1294_tumor_line_2" ]

        if idx_group_tmp == 5:  
            list_group_test_mice = [ "sts_1289_tumor_line_1", "sts_1289_tumor_line_2", "sts_1293_tumor_line_1", "sts_1293_tumor_line_2" ]

        if idx_group_tmp == 6:  
            list_group_test_mice = [ "sts_1288_tumor_line_1", "sts_1288_tumor_line_2", "sts_1266_tumor_line_2" ]

        if idx_group_tmp == 7:  
            list_group_test_mice = [ "sts_1288_tumor_line_1", "sts_1288_tumor_line_2", "sts_1294_tumor_line_1", "sts_1294_tumor_line_2" ]

        if idx_group_tmp == 8:  
            list_group_test_mice = [ "sts_1288_tumor_line_1", "sts_1288_tumor_line_2", "sts_1293_tumor_line_1", "sts_1293_tumor_line_2" ]

        if idx_group_tmp == 9:  
            list_group_test_mice = [ "sts_1288_tumor_line_1", "sts_1288_tumor_line_2", "sts_1289_tumor_line_1", "sts_1289_tumor_line_2" ]

        # statistics 
        list_of_infer        = []
        list_of_label        = []
        list_of_acc          = []
        list_of_tpr          = []
        list_of_tnr          = [] 
        list_of_fpr          = []
        list_of_fnr          = [] 
        list_of_cnr          = []

        list_of_y_true_all   = [] 
        list_of_y_predict_all = []

        count_acc_tmp        = 0
        count_tp_tmp         = 0 
        count_tn_tmp         = 0
        count_fp_tmp         = 0 
        count_fn_tmp         = 0 
        count_all_tmp        = 0 

        for name_of_mice_tmp in list_group_test_mice:

            # label-gt = high-scanning data 
            idx_scan_high_or_label              = "high_scan"

            # save the label 
            label_line_tmp = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_label.npy" )

            # inference 
            para_input_tmp                          = {} 
            para_input_tmp["mean_spectra"]          = self._mean_data_tmp
            para_input_tmp["std_spectra"]           = self._std_data_tmp
            para_input_tmp["loss_mode"]             = "NLLLoss"
            para_input_tmp["data_npy_stich_wv_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_wv.npy" )
            para_input_tmp["data_npy_stich_sp_raw"] = np.load( self._global_data_histopathology + name_of_mice_tmp + "_" + idx_scan_high_or_label + "_sp.npy" )
            infer_line_tmp                          = self._unit_mlp_infer_class.infer_npy_data_check( para_input = para_input_tmp )

            # summary 
            res_infer_tmp = np.asanyarray( infer_line_tmp ).astype(np.uint8).tolist()
            res_label_tmp = label_line_tmp.tolist()

            list_of_infer = list_of_infer + res_infer_tmp
            list_of_label = list_of_label + res_label_tmp

            # statistics
            # TODO: implement the functions
            res_acc_sum, count_tp, count_tn, count_fp, count_fn, count_all, list_of_y_true, list_of_y_predict = statistics_from_inference(res_inference=res_infer_tmp, res_label=res_label_tmp)
            count_acc_tmp += res_acc_sum
            count_tp_tmp  += count_tp
            count_tn_tmp  += count_tn
            count_fp_tmp  += count_fp
            count_fn_tmp  += count_fn
            count_all_tmp += count_all

            list_of_y_true_all += list_of_y_true
            list_of_y_predict_all += list_of_y_predict

        accr = np.round( count_acc_tmp / count_all_tmp, 2 )
        tpr  = np.round( count_tp_tmp / ( count_tp_tmp + count_fn_tmp ), 2 )
        tnr  = np.round( count_tn_tmp / ( count_tn_tmp + count_fp_tmp ), 2 ) 
        fpr  = np.round( count_fp_tmp / ( count_fp_tmp + count_tn_tmp ), 2 )
        fnr  = np.round( count_fn_tmp / ( count_tp_tmp + count_fn_tmp ), 2 ) 

        accr = count_acc_tmp
        tpr  = count_tp_tmp
        tnr  = count_tn_tmp
        fpr  = count_fp_tmp
        fnr  = count_fn_tmp

        list_of_acc.append( accr ) 
        list_of_tpr.append(  tpr )
        list_of_tnr.append(  tnr  )
        list_of_fpr.append(  fpr  )
        list_of_fnr.append(  fnr )
        list_of_cnr.append( count_all_tmp )

        print("list_of_infer = ", list_of_infer)
        print("list_of_label = ", list_of_label)
        print("list_of_tp = ", list_of_tpr)
        print("list_of_tn = ", list_of_tnr)
        print("list_of_fp = ", list_of_fpr)
        print("list_of_fn = ", list_of_fnr)

        # # vis with the boxplot (this is im)
        # # boxplot visualization 
        # data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
        # print("data = ", data.shape)
        # exit()

        # sns.boxplot(data = list_of_infer)
        # plt.xlim(0, 3)  # Shift the x-axis limits
        # plt.show()

        return list_of_acc, list_of_tpr, list_of_tnr, list_of_fpr, list_of_fnr, list_of_cnr, list_of_y_true_all, list_of_y_predict_all

    def inference_sts_three_to_two(self):
        """generalized system variables = testing with the pathology labels"""

        # define the global 
        self._global_data_histopathology    = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_three_to_two_sts/dataset_histopathology/"

        # test-2: mlp-inference agent
        num_of_combine  = 10

        # initialization 
        list_of_acc_all                     = []
        list_of_tp_all                      = []
        list_of_tn_all                      = []
        list_of_fp_all                      = []
        list_of_fn_all                      = []
        list_of_count_all                   = []
        list_of_y_true_all_global           = [] 
        list_of_y_predict_all_global        = []
        list_of_accuracy                    = []
        list_of_precision                   = []
        list_of_recall                      = []
        list_of_f1                          = []
        list_of_specificity                 = []
        list_of_tp_all_check                = []
        list_of_tn_all_check                = []
        list_of_fp_all_check                = []
        list_of_fn_all_check                = []
        list_of_num_train_and_val           = []
        list_of_num_test_with_label         = []

        """group-index = [ [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], 
                           [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]  ]
        """
        list_of_group_idx_combine           = [ [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], 
                                                [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]  ]
        list_of_group_idx_var               = []
        
        for idx_test in range( num_of_combine ):
    
            # update the group index 
            idx_group_use   = idx_test

            # obtain the group name 
            var_idx         = list_of_group_idx_combine[idx_test]
            var_tmp         = "STS-" + str(var_idx[0]) + " + STS-" + str(var_idx[1]) + " + STS-" + str(var_idx[2])
            list_of_group_idx_var.append( var_tmp )

            # init the inference model 
            num_of_train_and_val = self.init_mlp_inference_model_sts_three_to_two(idx_group = idx_group_use)
            list_of_num_train_and_val.append( num_of_train_and_val )

            # develop the cross-validation method
            list_of_acc, list_of_tp, list_of_tn, list_of_fp, list_of_fn, list_of_count, list_of_y_true_all, list_of_y_predict_all = self.mlp_inference_cross_validation_sts_three_to_two(idx_group_tmp = idx_group_use) 
            list_of_y_true_all_global       = list_of_y_true_all_global + list_of_y_true_all
            list_of_y_predict_all_global    = list_of_y_predict_all_global + list_of_y_predict_all
            list_of_num_test_with_label.append( len(list_of_y_predict_all) )
            
            # update the accuacy 
            list_of_acc_all     = list_of_acc_all   + list_of_acc 
            list_of_tp_all      = list_of_tp_all    + list_of_tp 
            list_of_tn_all      = list_of_tn_all    + list_of_tn
            list_of_fp_all      = list_of_fp_all    + list_of_fp
            list_of_fn_all      = list_of_fn_all    + list_of_fn
            list_of_count_all   = list_of_count_all + list_of_count

            # update the statistics + calculate confusion matrix
            y_true_flip     = flipping_label( list_of_y_true_all )
            y_predict_flip  = flipping_label( list_of_y_predict_all )

            # calculation from the built-in algorithms
            # print(classification_report( list_of_y_true_all, list_of_y_predict_all))
            
            # tp, fn, fp, tn  = confusion_matrix( y_true_flip, y_predict_flip ).ravel()
            tn, fp, fn, tp      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()

            accuracy        = accuracy_score( y_true_flip, y_predict_flip )
            precision       = precision_score( y_true_flip, y_predict_flip )
            recall          = recall_score( y_true_flip, y_predict_flip)
            f1              = f1_score( y_true_flip, y_predict_flip)
            specificity     = tn / (tn + fp)
            
            # checking 
            # the label is flipping towards the entire study.
            # print("checking = ")
            # print("list_of_tp = ", list_of_tp)
            # print("list_of_tn = ", list_of_tn)
            # print("list_of_fp = ", list_of_fp)
            # print("list_of_fn = ", list_of_fn)
            # print("tp = ", tp)
            # print("fp = ", fp)
            # print("tn = ", tn)
            # print("fn = ", fn)
            # exit() 

            list_of_tp_all_check.append( tp )
            list_of_tn_all_check.append( tn )
            list_of_fp_all_check.append( fp )
            list_of_fn_all_check.append( fn )
            list_of_accuracy.append( round( accuracy, 2) )
            list_of_precision.append( round( precision, 2) )
            list_of_recall.append( round( recall, 2) )
            list_of_f1.append( round( f1, 2) )
            list_of_specificity.append( round( specificity, 2) )

        # summary of the complete statistics 
        print("list_of_tp_all_check = ", list_of_tp_all_check)
        print("list_of_tn_all_check = ", list_of_tn_all_check)
        print("list_of_fp_all_check = ", list_of_fp_all_check)
        print("list_of_fn_all_check = ", list_of_fn_all_check)
        print("list_of_accuracy = ", list_of_accuracy)
        print("list_of_precision = ", list_of_precision)
        print("list_of_recall = ", list_of_recall)
        print("list_of_f1 = ", list_of_f1)
        print("list_of_specificity = ", list_of_specificity)
        print("list_of_count_all = ", list_of_count_all)
        print("list_of_group_idx_var = ", list_of_group_idx_var)
        print("list_of_num_train_and_val = ", list_of_num_train_and_val)
        print("list_of_num_test_with_label = ", list_of_num_test_with_label)

        # latex with the dataframe
        df_tmp                     = pd.DataFrame()
        df_tmp['Group index']      = list_of_group_idx_var
        df_tmp['train + val']      = list_of_num_train_and_val
        df_tmp['Label']            = list_of_num_test_with_label
        df_tmp['TP']               = list_of_tp_all_check
        df_tmp['TN']               = list_of_tn_all_check
        df_tmp['FP']               = list_of_fp_all_check
        df_tmp['FN']               = list_of_fn_all_check
        df_tmp['Accuracy']         = list_of_accuracy
        df_tmp['Accuracy']         = df_tmp['Accuracy'].apply(lambda x: f'{x:.2f}')
        df_tmp['Precision']        = list_of_precision
        df_tmp['Precision']        = df_tmp['Precision'].apply(lambda x: f'{x:.2f}')
        df_tmp['Recall']           = list_of_recall
        df_tmp['Recall']           = df_tmp['Recall'].apply(lambda x: f'{x:.2f}')
        df_tmp['F1-score']         = list_of_f1
        df_tmp['F1-score']         = df_tmp['F1-score'].apply(lambda x: f'{x:.2f}')
        df_tmp['Specificity']      = list_of_specificity
        df_tmp['Specificity']      = df_tmp['Specificity'].apply(lambda x: f'{x:.2f}')
        latex_tmp                  = df_tmp.to_latex(index=False)
        print("latex_tmp = ", latex_tmp)

    def inference_os_two_to_one(self):  
        """a = 1"""

        self._global_data_histopathology = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_two_to_one_os/dataset_histopathology/"

        # test-2: mlp-inference agent
        num_of_combine               = 3

        # initialization 
        list_of_acc_all              = []
        list_of_tp_all               = []
        list_of_tn_all               = []
        list_of_fp_all               = []
        list_of_fn_all               = []
        list_of_count_all            = []
        list_of_y_true_all_global    = [] 
        list_of_y_predict_all_global = []
        list_of_accuracy             = []
        list_of_precision            = []
        list_of_recall               = []
        list_of_f1                   = []
        list_of_specificity          = []
        list_of_tp_all_check         = [] 
        list_of_tn_all_check         = [] 
        list_of_fp_all_check         = [] 
        list_of_fn_all_check         = []
        list_of_num_train_and_val    = []
        list_of_num_test_with_label  = []

        """group-index """
        list_of_group_idx_combine           = [ [1, 2], [1, 3], [2, 3] ]
        list_of_group_idx_var               = []

        for idx_test in range( num_of_combine ):
    
            # update the group index 
            idx_group_use   = idx_test

            # obtain the group name 
            var_idx         = list_of_group_idx_combine[idx_test]
            var_tmp         = "OS-" + str(var_idx[0]) + " + OS-" + str(var_idx[1]) 
            list_of_group_idx_var.append( var_tmp )

            # init the inference model 
            num_of_train_and_val = self.init_mlp_inference_model_os_two_to_one(idx_group = idx_group_use)
            list_of_num_train_and_val.append( num_of_train_and_val )
            
            # develop the cross-validation method
            list_of_acc, list_of_tp, list_of_tn, list_of_fp, list_of_fn, list_of_count, list_of_y_true_all, list_of_y_predict_all = self.mlp_inference_cross_validation_os_two_to_one(idx_group_tmp = idx_group_use ) 
            list_of_y_true_all_global = list_of_y_true_all_global + list_of_y_true_all
            list_of_y_predict_all_global = list_of_y_predict_all_global + list_of_y_predict_all
            list_of_num_test_with_label.append( len(list_of_y_predict_all) )

            # update the accuacy 
            list_of_acc_all     = list_of_acc_all + list_of_acc 
            list_of_tp_all      = list_of_tp_all  + list_of_tp 
            list_of_tn_all      = list_of_tn_all  + list_of_tn
            list_of_fp_all      = list_of_fp_all  + list_of_fp
            list_of_fn_all      = list_of_fn_all  + list_of_fn
            list_of_count_all   = list_of_count_all + list_of_count

            # update the statistics
            # Calculate confusion matrix
            y_true_flip         = flipping_label( list_of_y_true_all )
            y_predict_flip      = flipping_label( list_of_y_predict_all )

            # print(classification_report( list_of_y_true_all, list_of_y_predict_all))
            
            # tp, fn, fp, tn      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()
            tn, fp, fn, tp      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()

            accuracy            = accuracy_score( y_true_flip, y_predict_flip )
            precision           = precision_score( y_true_flip, y_predict_flip )
            recall              = recall_score( y_true_flip, y_predict_flip)
            f1                  = f1_score( y_true_flip, y_predict_flip)
            specificity         = tn / (tn + fp)
            
            list_of_accuracy.append( round( accuracy, 2) )
            list_of_precision.append( round( precision, 2) )
            list_of_recall.append( round( recall, 2) )
            list_of_f1.append( round( f1, 2) )
            list_of_specificity.append( round( specificity, 2) )

            list_of_tp_all_check.append( tp )
            list_of_tn_all_check.append( tn )
            list_of_fp_all_check.append( fp )
            list_of_fn_all_check.append( fn )

            print("tp = ", tp)
            print("tn = ", tn)
            print("fp = ", fp)
            print("fn = ", fn)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Specificity:", specificity)

        # summary of the complete statistics 
        print("list_of_tp_all_check = ", list_of_tp_all_check)
        print("list_of_tn_all_check = ", list_of_tn_all_check)
        print("list_of_fp_all_check = ", list_of_fp_all_check)
        print("list_of_fn_all_check = ", list_of_fn_all_check)
        print("list_of_accuracy = ", list_of_accuracy)
        print("list_of_precision = ", list_of_precision)
        print("list_of_recall = ", list_of_recall)
        print("list_of_f1 = ", list_of_f1)
        print("list_of_specificity = ", list_of_specificity)
        print("list_of_count_all = ", list_of_count_all)
        print("list_of_group_idx_var = ", list_of_group_idx_var)
        print("list_of_num_train_and_val = ", list_of_num_train_and_val)
        print("list_of_num_test_with_label = ", list_of_num_test_with_label)

        # latex with the dataframe
        df_tmp                     = pd.DataFrame()
        df_tmp['Group index']      = list_of_group_idx_var
        df_tmp['train + val']      = list_of_num_train_and_val
        df_tmp['Label']            = list_of_num_test_with_label
        df_tmp['TP']               = list_of_tp_all_check
        df_tmp['TN']               = list_of_tn_all_check
        df_tmp['FP']               = list_of_fp_all_check
        df_tmp['FN']               = list_of_fn_all_check
        df_tmp['Accuracy']         = list_of_accuracy
        df_tmp['Accuracy']         = df_tmp['Accuracy'].apply(lambda x: f'{x:.2f}')
        df_tmp['Precision']        = list_of_precision
        df_tmp['Precision']        = df_tmp['Precision'].apply(lambda x: f'{x:.2f}')
        df_tmp['Recall']           = list_of_recall
        df_tmp['Recall']           = df_tmp['Recall'].apply(lambda x: f'{x:.2f}')
        df_tmp['F1-score']         = list_of_f1
        df_tmp['F1-score']         = df_tmp['F1-score'].apply(lambda x: f'{x:.2f}')
        df_tmp['Specificity']      = list_of_specificity
        df_tmp['Specificity']      = df_tmp['Specificity'].apply(lambda x: f'{x:.2f}')
        latex_tmp                  = df_tmp.to_latex(index=False)
        print("latex_tmp = ", latex_tmp)
        
    def inference_sts_four_to_one(self): 

        self._global_data_histopathology    = "./exp_mice_histopathology/mouse_mlp_study/result_mlp_cross_validation_four_to_one_sts/dataset_histopathology/"

        # initialization 
        list_of_acc_all                     = []
        list_of_tp_all                      = []
        list_of_tn_all                      = []
        list_of_fp_all                      = []
        list_of_fn_all                      = []
        list_of_count_all                   = []
        list_of_y_true_all_global           = [] 
        list_of_y_predict_all_global        = []
        list_of_accuracy                    = []
        list_of_precision                   = []
        list_of_recall                      = []
        list_of_f1                          = []
        list_of_specificity                 = []
        list_of_tp_all_check                = []
        list_of_tn_all_check                = []
        list_of_fp_all_check                = []
        list_of_fn_all_check                = []
        list_of_num_train_and_val           = []
        list_of_num_test_with_label         = []

        """group-index """
        list_of_group_idx_combine           = [ [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5] ]
        list_of_group_idx_var               = []
        num_of_combine                      = len( list_of_group_idx_combine ) 
        
        for idx_test in range( num_of_combine ):
    
            # update the group index 
            idx_group_use   = idx_test

            # obtain the group name 
            var_idx         = list_of_group_idx_combine[idx_test]
            var_tmp         = "STS-" + str(var_idx[0]) + " + STS-" + str(var_idx[1]) + " + STS-" + str(var_idx[2]) + " + STS-" + str(var_idx[3])
            list_of_group_idx_var.append( var_tmp )

            # init the inference model 
            num_of_train_and_val = self.init_mlp_inference_model_four_to_one(idx_group = idx_group_use)
            list_of_num_train_and_val.append( num_of_train_and_val )

            # develop the cross-validation method
            list_of_acc, list_of_tp, list_of_tn, list_of_fp, list_of_fn, list_of_count, list_of_y_true_all, list_of_y_predict_all = self.mlp_inference_cross_validation_four_to_one(idx_group_tmp = idx_group_use ) 
            list_of_y_true_all_global = list_of_y_true_all_global + list_of_y_true_all
            list_of_y_predict_all_global = list_of_y_predict_all_global + list_of_y_predict_all
            list_of_num_test_with_label.append( len( list_of_y_true_all ) )

            # update the accuacy 
            list_of_acc_all     = list_of_acc_all + list_of_acc 
            list_of_tp_all      = list_of_tp_all  + list_of_tp 
            list_of_tn_all      = list_of_tn_all  + list_of_tn
            list_of_fp_all      = list_of_fp_all  + list_of_fp
            list_of_fn_all      = list_of_fn_all  + list_of_fn
            list_of_count_all   = list_of_count_all + list_of_count

            # update the statistics
            # Calculate confusion matrix
            y_true_flip         = flipping_label( list_of_y_true_all )
            y_predict_flip      = flipping_label( list_of_y_predict_all )

            # print(classification_report( list_of_y_true_all, list_of_y_predict_all))
            # tp, fn, fp, tn      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()
            # since we flip the labels and thus we need to generate different results. 
            tn, fp, fn, tp      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()
            accuracy            = accuracy_score( y_true_flip, y_predict_flip )
            precision           = precision_score( y_true_flip, y_predict_flip )
            recall              = recall_score( y_true_flip, y_predict_flip)
            f1                  = f1_score( y_true_flip, y_predict_flip)
            specificity         = tn / (tn + fp)

            # checking the results 
            print("y_true_raw =  ", list_of_y_true_all)
            print("y_pred_raw =  ", list_of_y_predict_all)
            print("y_true_flip = ", y_true_flip)
            print("y_pred_flip = ", y_predict_flip)
            print("tp = ", tp)
            print("list_of_tp = ", list_of_tp)
            print("tn = ", tn)
            print("list_of_tn = ", list_of_tn)
            print("fp = ", fp)
            print("list_of_fp = ", list_of_fp)
            print("fn = ", fn)
            print("list_of_fn = ", list_of_fn)

            # checking  
            print("data_arry_predicted_output = ", list_of_y_predict_all)
            print("label_test_local = ", list_of_y_true_all)
            print("y_predict_flip = ", y_predict_flip)
            print("y_true_flip = ", y_true_flip)
            print("tp = ", tp)
            print("count_tp = ", list_of_tp)
            print("tn = ", tn)
            print("count_tn = ", list_of_tn)
            print("fp = ", fp)
            print("count_fp = ", list_of_fp)
            print("fn = ", fn)
            print("count_fn = ", list_of_fn)

            print("calculated accuracy = ", accuracy)
            accuracy_check = ( tp + tn ) / ( tp + tn + fp + fn )
            print("manual accuracy = ", accuracy_check)

            print("calculated precision = ", precision)
            precision_check = tp / ( tp + fp )
            print("manual precision = ", precision_check)

            print("calculated recall = ", recall)
            recall_check = tp / ( tp + fn )
            print("manual recall = ", recall_check)

            print("calculated f1 = ", f1)
            f1_check = 2 * ( precision_check * recall_check ) / ( precision_check + recall_check )
            print("manual f1 = ", f1_check)

            # input("checking")

            """checking of the calculation results 
            """
            print("tp = ", tp)
            print("fn = ", fn)
            print("fp = ", fp)
            print("tn = ", tn)

            print("list_of_tp = ", list_of_tp)
            print("list_of_tn = ", list_of_tn)
            print("list_of_fp = ", list_of_fp)
            print("list_of_fn = ", list_of_fn)

            # exit()

            list_of_accuracy.append( round( accuracy, 2) )
            list_of_precision.append( round( precision, 2) )
            list_of_recall.append( round( recall, 2) )
            list_of_f1.append( round( f1, 2) )
            list_of_specificity.append( round( specificity, 2) )

            list_of_tp_all_check.append( tp )
            list_of_tn_all_check.append( tn )
            list_of_fp_all_check.append( fp )
            list_of_fn_all_check.append( fn )

        # summary of the complete statistics 
        print("list_of_tp_all_check = ", list_of_tp_all_check)
        print("list_of_tn_all_check = ", list_of_tn_all_check)
        print("list_of_fp_all_check = ", list_of_fp_all_check)
        print("list_of_fn_all_check = ", list_of_fn_all_check)
        print("list_of_accuracy = ", list_of_accuracy)
        print("list_of_precision = ", list_of_precision)
        print("list_of_recall = ", list_of_recall)
        print("list_of_f1 = ", list_of_f1)
        print("list_of_specificity = ", list_of_specificity)
        print("list_of_count_all = ", list_of_count_all)
        print("list_of_group_idx_var = ", list_of_group_idx_var)
        print("list_of_num_train_and_val = ", list_of_num_train_and_val)
        print("list_of_num_test_with_label = ", list_of_num_test_with_label)

        # latex with the dataframe
        df_tmp                     = pd.DataFrame()
        df_tmp['Group index']      = list_of_group_idx_var
        df_tmp['train + val']      = list_of_num_train_and_val
        df_tmp['Label']            = list_of_num_test_with_label
        df_tmp['TP']               = list_of_tp_all_check
        df_tmp['TN']               = list_of_tn_all_check
        df_tmp['FP']               = list_of_fp_all_check
        df_tmp['FN']               = list_of_fn_all_check
        df_tmp['Accuracy']         = list_of_accuracy
        df_tmp['Accuracy']         = df_tmp['Accuracy'].apply(lambda x: f'{x:.2f}')
        df_tmp['Precision']        = list_of_precision
        df_tmp['Precision']        = df_tmp['Precision'].apply(lambda x: f'{x:.2f}')
        df_tmp['Recall']           = list_of_recall
        df_tmp['Recall']           = df_tmp['Recall'].apply(lambda x: f'{x:.2f}')
        df_tmp['F1-score']         = list_of_f1
        df_tmp['F1-score']         = df_tmp['F1-score'].apply(lambda x: f'{x:.2f}')
        df_tmp['Specificity']      = list_of_specificity
        df_tmp['Specificity']      = df_tmp['Specificity'].apply(lambda x: f'{x:.2f}')
        latex_tmp                  = df_tmp.to_latex(index=False)
        print("latex_tmp = ", latex_tmp)

def statistics_from_inference(res_inference = [], res_label = []): 

    """inference VS ground truth
    # statistics 
    # false positive 
    # false negative 
    # true positivee
    # true negative 
    # accuracy
    # precision 
    # recall
    # F1 score

    we define the positive case as the tumorous region
    we define the negative case as the healthy region

    """

    # True-positive + True-negative + False-positive + False-negativee
    # TP + TN + FP + FN 

    # accuracy 
    res_count_all       = len( res_inference )  
    res_acc_sum         = 0
    count_tp            = 0 
    count_tn            = 0 
    count_fp            = 0
    count_fn            = 0

    # summarize the list of prediction
    list_of_y_true      = [] 
    list_of_y_predict   = [] 

    for idx, item in enumerate( res_label ):
        
        # nan case
        # print("item = ", item)
        if np.isnan( item ):
            res_count_all = res_count_all - 1
            continue

        # accuracy of the system  
        elif int( res_inference[idx] ) == int( res_label[idx] ):
            res_acc_sum += 1

        """statistics"""

        # true positive 
        # prediction == tumor == actual 
        if ( int( res_inference[idx] ) == int( res_label[idx] ) ) and ( int( res_label[idx] ) == 0 ) :
            count_tp += 1

        # true negative 
        # prediction == healthy == actual
        elif ( int( res_inference[idx] ) == int( res_label[idx] ) ) and ( int( res_label[idx] ) == 1 ):
            count_tn += 1

        # false positive
        # predition == tumor + actual == healthy
        elif int( res_inference[idx] ) == 0 and int( res_label[idx] ) == 1:
            count_fp += 1

        # false negative 
        # predition == healthy + actual == tumor
        elif int( res_inference[idx] ) == 1 and int( res_label[idx] ) == 0:
            count_fn += 1

        # print("count_tp = ", count_tp)

        # summarize the results
        list_of_y_true.append( int( res_label[idx] ) )
        list_of_y_predict.append( int( res_inference[idx] ) )

    return res_acc_sum, count_tp, count_tn, count_fp, count_fn, res_count_all, list_of_y_true, list_of_y_predict

def flipping_label( y_ary ):

    y_ary_flip = []

    for item in y_ary: 

        if item == 0: 
            y_ary_flip.append( 1 )
        else: 
            y_ary_flip.append( 0 )

    return y_ary_flip

def get_permutations_no_repeat(sequence):
  """
  Generates all permutations of a sequence without repeating elements.

  Args:
    sequence: The input sequence (e.g., string, list, tuple).

  Returns:
    A list of tuples, where each tuple is a permutation of the input sequence.
  """
  return list(itertools.permutations(sequence))

if __name__ == "__main__":

    # define the class
    test_class = test_internal()

    """training: sts three-to-two model"""
    test_class.init_mlp_sts_three_to_two_setting()
    test_class.init_mlp_train_model()
    test_class.mlp_train_cross_validation_three_to_two()
    test_class.inference_sts_three_to_two()
    exit() 

    """training: sts four-to-one model"""
    test_class.init_mlp_sts_four_to_one_setting()
    test_class.init_mlp_train_model()
    test_class.mlp_train_cross_validation_four_to_one()
    test_class.inference_sts_four_to_one()
    # exit()

    """training os two-to-one model"""
    test_class.init_mlp_os_two_to_one_setting()
    test_class.init_mlp_train_model()
    test_class.mlp_train_cross_validation_two_to_one()
    test_class.inference_os_two_to_one()

