
import os 
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import itertools
from itertools import permutations
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# pytorch related 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append("./utility/")

# modules
import ml_model_train_edit_org 
import ml_inference_edit_org 
from mlp_train_and_inference_cross_validation_tumor_histopathology import statistics_from_inference

# define the random seed 
random.seed(42)
np.random.seed(42)
torch.manual_seed(0)

class mlp_tumor_degree_analysis():

    def __init__(self):

        # tumormap model agent
        # self._tumormap_database_agent = tumormap_database_mlp_edit_org.tumormap_database_mlp()

        # # tumormap inference model agent
        self._unit_mlp_infer_class = ml_inference_edit_org.mlp_inference_unit_test()

    def init_sts_inference(self, idx_group = [] ):

        # unit-1: test the inference class 
        para_inference_input                   = {}
        para_inference_input["model_name"]     = "tumormap"
        para_inference_input["loss_mode"]      = "NLLLoss"
        para_inference_input["model_dict"]     = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_sts/pth_res/"
        para_inference_input["log_instance"]   = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]   = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )
        # checking:
        # self._mlp_inference_model.eval()
        
        # unit-2-a: train mlp 
        # loading this class aims to reload this function: self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        """global parameter settings 
            self._val_batchsize             = 16
            self._val_numepoch              = 150
            self._lr                        = 0.001
            self._idx_use_gpu               = 0
            self._is_use_cpu                = False
        """
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # mean and std from the training dataset
        self._unit_mlp_infer_class._path_global_dataset = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_sts/dataset/"
        path_data_use_for_training             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                 = np.load( path_data_use_for_training )
        self._data_use_for_train_and_val       = data_npy_train_and_val
        mean_data_tmp, std_data_tmp            = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                    = mean_data_tmp
        self._std_data_tmp                     = std_data_tmp

        # load the testing dataset 
        path_data_use_for_testing              = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_test.npy"
        path_label_use_for_testing             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_label_test.npy"
        self._data_npy_test                    = np.load( path_data_use_for_testing )
        self._label_npy_test                   = np.load( path_label_use_for_testing )

    def init_os_inference(self, idx_group = [] ):
        """perform the inference model"""

        # load the training models
        # unit-1: test the inference class 
        # output: self._mlp_inference_model
        # result_mlp_cross_validation_six_to_two_os
        para_inference_input                   = {}
        para_inference_input["model_name"]     = "tumormap"
        para_inference_input["loss_mode"]      = "NLLLoss"
        para_inference_input["model_dict"]     = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/pth_res/"
        para_inference_input["log_instance"]   = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]   = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )
        
        # check: inference from the CPU -> mimic the deployment to keep consistency.
        # unit-2-a: train mlp 
        # output: self._mlp_train_model
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # training and validation 
        # inference for a separated dataset (inference)
        # mean and std from the training dataset
        # result_mlp_cross_validation_six_to_two_os
        self._unit_mlp_infer_class._path_global_dataset = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/dataset/"
        path_data_use_for_training             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                 = np.load( path_data_use_for_training )
        self._data_use_for_train_and_val       = data_npy_train_and_val
        mean_data_tmp, std_data_tmp            = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                    = mean_data_tmp
        self._std_data_tmp                     = std_data_tmp

        # load the testing dataset 
        path_data_use_for_testing              = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_test.npy"
        path_label_use_for_testing             = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_label_test.npy"
        self._data_npy_test                    = np.load( path_data_use_for_testing )
        self._label_npy_test                   = np.load( path_label_use_for_testing )

        # print("path_data_use_for_training = ", path_data_use_for_training)
        # print("mean_data_tmp = ", mean_data_tmp)
        # print("std_data_tmp  = ", std_data_tmp)
        # input("check mean and std")
        # exit()

    def check_combination_group(self, items_main = [], list_comb_main = [], k = 0 ):
        """
        os_tumor:
            1. os_1241
            2. os_1243
            3. os_1244
            4. os_1273
            5. os_1274
            6. os_1276
            7. os_1296
        """
        combinations_list = list(itertools.combinations(items_main, k))
        traj_of_training = []
        traj_of_testing  = []
        for i in range(len(combinations_list)):
            comb_training = [] 
            comb_testing  = []
            for item_check in list_comb_main:
                if item_check not in combinations_list[i]:
                    comb_testing.append( int( item_check ) )
                else:
                    comb_training.append( int( item_check ) )
            traj_of_training.append(comb_training)
            traj_of_testing.append(comb_testing)

        return traj_of_training, traj_of_testing

    def sts_mlp_train(self): 
        """sts-model-only
            result_mlp_cross_validation_six_to_two_sts = sts + 6 : 2
        """

        # step-1: select the combination group
        num_of_item     = 6
        num_of_select   = 4
        items_main      = range(0, num_of_item)  
        list_comb_main  = np.linspace( 0, num_of_item - 1, num_of_item ) 
        k               = num_of_select
        traj_of_training, traj_of_testing = utility_class.check_combination_group( items_main = items_main, list_comb_main = list_comb_main, k = k )
        
        # path uniquely for the data and models
        path_local_for_mlp = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_sts"

        # training + validation datasets
        for idx_comb in range(len(traj_of_training) ):

            if idx_comb > 0: 
                continue

            self.path_save_model        = path_local_for_mlp + "/pth_res/"
            self.path_save_acc          = path_local_for_mlp + "/acc_res/"
            self.path_save_loss         = path_local_for_mlp + "/loss_res/"
            self.path_dataset           = path_local_for_mlp + "/dataset/"
            self.model_name             = "tumormap"
            self.log_instance           = "group_idx_" + str( idx_comb )

            # load the dataset
            ary_sp_train    = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_data_train_and_val.npy" )
            ary_label_train = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_label_train_and_val.npy" )

            # model training
            # define the model architectures
            val_batchsize   = 16
            val_numepoch    = 150
            parse = {
                "gpu": 0,
                "use_cpu": False, # True
                "batch_size": val_batchsize,            # 16,
                "num_epochs": val_numepoch,             # 150,
                "lr": 0.001,
                "model_name": "tumor_map",
                "path_save_model": [], 
                "log_dir": "./data_mlp/logs/",
                "flag": "train",
                "tumor_type": "STS",                    # "STS" or "OS"
                "log_instance": "sts_official_cross",         # "sts_official_1" or "os_official_1"
                "loss_mode": "NLLLoss"                  # NLLLoss or cross_entropy
                    }
            print("log_instance = ", parse["log_instance"])
            print("tumor_type = ", parse["tumor_type"])
            print("val_batchsize = ", parse["batch_size"])
            print("loss_mode = ", parse["loss_mode"])
            print("num_epochs = ", parse["num_epochs"])
            self._tumormap_agent   = ml_model_train_edit_org .TumorMAP( parse = parse )

            # mean and std of dataset
            # normalize dataset using mean and std (z score normalization)
            database_mean, database_std = self._tumormap_agent.dataset_normalize( ary_sp_train )
            ary_sp_train                = (ary_sp_train - database_mean) / database_std

            # train and validation
            x_train, x_val, y_train, y_val = train_test_split( ary_sp_train, ary_label_train, test_size = 0.3, random_state = 42, stratify = ary_label_train )

            # build the model architecture
            model, criterion, optimizer = self._tumormap_agent.build_model()
            scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            early_stopper               = ml_model_train_edit_org .EarlyStopping(patience=5, delta = 0.5)

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

            # # save the model
            torch.save(model.state_dict(), self.path_save_model + self.model_name + "_" + self.log_instance + ".pth")
            print("Model saved successfully")

            self._tumormap_agent.writer.close()

            # loss curve
            size_font_fig = 20  
            plt.plot( train_loss_list, linestyle='--', marker='o', color='r', label = "Training")
            plt.plot( val_loss_list, linestyle='--', marker='o', color='g', label = "Validation")
            plt.xlabel("Epoch", fontsize = size_font_fig )
            plt.ylabel("MSE loss values", fontsize = size_font_fig )
            plt.title("Loss curve" + "(group-idx-" + str( idx_comb ) + ")", fontsize = size_font_fig )
            plt.xticks( fontsize = size_font_fig )
            plt.yticks( fontsize = size_font_fig )
            plt.grid()
            plt.legend( fontsize = size_font_fig )
            plt.ylim([0.0, 3.0])
            plt.savefig(self.path_save_loss + self.log_instance + "_loss.png", bbox_inches='tight')
            plt.clf()
            plt.cla()

            # Accuracy curve
            plt.plot( train_acc_list, linestyle='--', marker='*', color='r', label='Training')
            plt.plot( val_acc_list, linestyle='--', marker='*', color='g', label='Validation')
            plt.xlabel("Epoch", fontsize = size_font_fig )
            plt.ylabel("Accuracy (unit: mm)", fontsize = size_font_fig )
            plt.title("Accuracy curve" + "(group-idx-" + str( idx_comb ) + ")", fontsize = size_font_fig )
            plt.xticks( fontsize = size_font_fig )
            plt.yticks( fontsize = size_font_fig )
            plt.ylim([0.0, 1.0])
            plt.grid()
            plt.legend()
            # plt.show()
            plt.legend( fontsize = size_font_fig )
            plt.savefig(self.path_save_acc + self.log_instance + "_acc.png", bbox_inches='tight')
            plt.clf()
            plt.cla()

            # # save the acc and loss
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_train_acc.npy", train_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_train_loss.npy", train_loss_list )
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_val_acc.npy", val_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_val_loss.npy", val_loss_list )

    def os_mlp_train(self):  
        """os-model-only
        checking == result_mlp_cross_validation_six_to_two_os = 6 : 2 
        """

        """step-1: selection of data combination"""
        num_of_item     = 6
        num_of_select   = 4
        items_main      = range(0, num_of_item)  
        list_comb_main  = np.linspace( 0, num_of_item - 1, num_of_item ) 
        k               = num_of_select
        traj_of_training, _ = utility_class.check_combination_group( items_main = items_main, list_comb_main = list_comb_main, k = k )
        
        # data related path
        path_local_for_mlp =  "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/"
        
        # training + validation datasets
        # number of (traj_of_training) = number of (traj_of_testing) thus we use the same loop
        for idx_comb in range(len(traj_of_training) ):

            if idx_comb > 0:
                break

            # prepare for the model testing and setting 
            self.path_save_model        = path_local_for_mlp + "pth_res/"
            self.path_save_acc          = path_local_for_mlp + "acc_res/"
            self.path_save_loss         = path_local_for_mlp + "loss_res/"
            self.path_dataset           = path_local_for_mlp + "dataset/"
            self.model_name             = "tumormap"
            self.log_instance           = "group_idx_" + str( idx_comb )

            # load the training and validation dataset
            ary_sp_train    = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_data_train_and_val.npy" )
            ary_label_train = np.load( self.path_dataset + self.model_name + "_" + self.log_instance + "_label_train_and_val.npy" )

            # model training
            # define the model architectures
            val_batchsize   = 16
            val_numepoch    = 150
            parse = {
                "gpu": 0,
                "use_cpu": False, 
                "batch_size": val_batchsize,          
                "num_epochs": val_numepoch,            
                "lr": 0.001,
                "model_name": "tumor_map",
                "path_save_model": [], 
                "log_dir": "./data_mlp/logs/",
                "flag": "train",
                "tumor_type": "OS",          
                "log_instance": "os_official_cross",       
                "loss_mode": "NLLLoss"                 
                    }
            print("log_instance = ", parse["log_instance"])
            print("tumor_type = ", parse["tumor_type"])
            print("val_batchsize = ", parse["batch_size"])
            print("loss_mode = ", parse["loss_mode"])
            print("num_epochs = ", parse["num_epochs"])
            self._tumormap_agent   = ml_model_train_edit_org .TumorMAP( parse = parse )

            # mean and std of dataset
            # normalize dataset using mean and std (z score normalization)
            database_mean, database_std = self._tumormap_agent.dataset_normalize( ary_sp_train )
            ary_sp_train                = (ary_sp_train - database_mean) / database_std

            # train and validation set 
            x_train, x_val, y_train, y_val = train_test_split( ary_sp_train, ary_label_train, test_size = 0.3, random_state = 42, stratify = ary_label_train)

            # build the model architecture
            model, criterion, optimizer = self._tumormap_agent.build_model()
            
            # mlp scheduler
            scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            # definition of an earlier stopper
            # object is a component used in PyTorch to halt the training of a neural network early if it stops making significant improvements, which prevents the model from overfitting. 
            early_stopper               = ml_model_train_edit_org.EarlyStopping(patience=5, delta = 0.5)
        
            # convert the data to tensor
            x_train_tensor  = torch.from_numpy(x_train).float()
            y_train_tensor  = torch.from_numpy(y_train).long()
            x_val_tensor    = torch.from_numpy(x_val).float()
            y_val_tensor    = torch.from_numpy(y_val).long()
            if self._tumormap_agent.use_cpu:
                # CPU operation 
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

            # officically prepare for the dataset 
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

            self._tumormap_agent.writer.close()

            # loss curve
            size_font_fig = 20  
            plt.plot( train_loss_list, linestyle='--', marker='o', color='r', label = "Training")
            plt.plot( val_loss_list, linestyle='--', marker='o', color='g', label = "Validation")
            plt.xlabel("Epoch", fontsize = size_font_fig )
            plt.ylabel("MSE loss values", fontsize = size_font_fig )
            plt.title("Loss curve" + "(group-idx-" + str( idx_comb ) + ")", fontsize = size_font_fig )
            plt.xticks( fontsize = size_font_fig )
            plt.yticks( fontsize = size_font_fig )
            plt.grid()
            plt.legend( fontsize = size_font_fig )
            plt.ylim([0.0, 3.0])
            plt.savefig(self.path_save_loss + self.log_instance + "_loss.png", bbox_inches='tight')
            plt.clf()
            plt.cla()

            # Accuracy curve
            plt.plot( train_acc_list, linestyle='--', marker='*', color='r', label='Training')
            plt.plot( val_acc_list, linestyle='--', marker='*', color='g', label='Validation')
            plt.xlabel("Epoch", fontsize = size_font_fig )
            plt.ylabel("Accuracy (unit: mm)", fontsize = size_font_fig )
            plt.title("Accuracy curve" + "(group-idx-" + str( idx_comb ) + ")", fontsize = size_font_fig )
            plt.xticks( fontsize = size_font_fig )
            plt.yticks( fontsize = size_font_fig )
            plt.ylim([0.0, 1.0])
            plt.grid()
            plt.legend()
            # plt.show()
            plt.legend( fontsize = size_font_fig )
            plt.savefig(self.path_save_acc + self.log_instance + "_acc.png", bbox_inches='tight')
            plt.clf()
            plt.cla()

            # # # save the acc and loss
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_train_acc.npy", train_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_train_loss.npy", train_loss_list )
            # np.save( self.path_save_acc + self.model_name + "_" + self.log_instance + "_val_acc.npy", val_acc_list )
            # np.save( self.path_save_loss + self.model_name + "_" + self.log_instance + "_val_loss.npy", val_loss_list )

    def sts_mlp_test(self): 
        """sts model testing"""

        path_statistics_save                = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_sts/"

        # test-2: mlp-inference agent
        num_of_combine                      = 15

        # initialization 
        list_of_count_all                   = []
        list_of_accuracy                    = []
        list_of_precision                   = []
        list_of_recall                      = []
        list_of_f1                          = []
        list_of_specificity                 = []
        list_of_tp_all_check                = []
        list_of_tn_all_check                = []
        list_of_fp_all_check                = []
        list_of_fn_all_check                = []

        list_of_train_and_val               = []
        list_of_test                        = []

        for idx_test in range( num_of_combine ):
    
            # update the group index 
            idx_group_use = idx_test

            # initialize the model 
            idx_group_tmp = idx_test
            self.init_sts_inference( idx_group = idx_group_tmp )

            # testing dataset
            data_test_local  = self._data_npy_test
            label_test_local = self._label_npy_test 
            
            # count the numebr of points
            num_of_train_and_val = self._data_use_for_train_and_val.shape[0]
            num_of_test          = self._data_npy_test.shape[0]

            # inference     
            data_arry_predicted_output = self._unit_mlp_infer_class.infer_npy_dataset( database_npy = data_test_local, 
                                                                                        mean_spectra = self._mean_data_tmp, 
                                                                                        std_spectra = self._std_data_tmp, 
                                                                                        loss_mode = "NLLLoss")
            # statistics
            res_acc_sum, count_tp, count_tn, count_fp, count_fn, count_all, list_of_y_true, list_of_y_predict = statistics_from_inference(res_inference = data_arry_predicted_output, 
                                                                                                                                          res_label = label_test_local)
            
            y_predict_flip  = flipping_label( data_arry_predicted_output )
            y_true_flip     = flipping_label(  label_test_local )
            tn, fp, fn, tp      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()

            accuracy        = accuracy_score( y_true_flip, y_predict_flip )
            precision       = precision_score( y_true_flip, y_predict_flip )
            recall          = recall_score( y_true_flip, y_predict_flip)
            f1              = f1_score( y_true_flip, y_predict_flip)
            specificity     = tn / (tn + fp)

            list_of_tp_all_check.append( tp )
            list_of_tn_all_check.append( tn )
            list_of_fp_all_check.append( fp )
            list_of_fn_all_check.append( fn )
            list_of_accuracy.append( np.around( accuracy, 2) )
            list_of_precision.append( np.around( precision, 2) )
            list_of_recall.append( np.around( recall, 2) )
            list_of_f1.append( np.around( f1, 2) )
            list_of_specificity.append( np.around( specificity, 2) )
            list_of_train_and_val.append(num_of_train_and_val)
            list_of_test.append(num_of_test)

            print("res_acc_sum = ", np.around ( res_acc_sum, 2 ) )
            print("count_tp = ", np.around ( count_tp, 2 ) )
            print("count_tn = ", np.around ( count_tn, 2 ) )
            print("count_fp = ", np.around ( count_fp, 2 ) )
            print("accuracy = ", np.around ( accuracy, 2 ) )
            print("precision = ", np.around ( precision, 2 ) )
            print("recall = ", np.around ( recall, 2) )
            print("f1 = ", np.around ( f1, 2 ) )
            print("specificity = ", np.around ( specificity, 2) )

            """checking"""  
            print("data_arry_predicted_output = ", data_arry_predicted_output)
            print("label_test_local = ", label_test_local)
            print("y_predict_flip = ", y_predict_flip)
            print("y_true_flip = ", y_true_flip)
            print("tp = ", tp)
            print("count_tp = ", count_tp)
            print("tn = ", tn)
            print("count_tn = ", count_tn)
            print("fp = ", fp)
            print("count_fp = ", count_fp)
            print("fn = ", fn)
            print("count_fn = ", count_fn)

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

        # summary
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
        
        # np.save( path_statistics_save + "res_train_count.npy", list_of_train_and_val )
        # np.save( path_statistics_save + "res_test_count.npy", list_of_test )
        # np.save( path_statistics_save + "res_tp_count.npy", list_of_tp_all_check )
        # np.save( path_statistics_save + "res_tn_count.npy", list_of_tn_all_check )
        # np.save( path_statistics_save + "res_fp_count.npy", list_of_fp_all_check )
        # np.save( path_statistics_save + "res_fn_count.npy", list_of_fn_all_check )
        # np.save( path_statistics_save + "res_accuracy.npy", list_of_accuracy )
        # np.save( path_statistics_save + "res_precision.npy", list_of_precision )
        # np.save( path_statistics_save + "res_recall.npy", list_of_recall )
        # np.save( path_statistics_save + "res_f1.npy", list_of_f1 )
        # np.save( path_statistics_save + "res_specificity.npy", list_of_specificity )

        data_pd = [ np.transpose(list_of_accuracy), np.transpose(list_of_precision), np.transpose(list_of_recall), np.transpose(list_of_f1), np.transpose(list_of_specificity) ]
        df = pd.DataFrame( data_pd )
        print("df = ", df)

    def os_mlp_test(self):
        """model testing for the testing dataset to evaluate the model performances"""

        """save the statistics to the local path"""
        path_statistics_save                  = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/"

        # test-2: mlp-inference agent
        # 6:4 ratios in terms of the combinations
        num_of_combine                      = 15

        # initialization 
        list_of_acc_all                     = []
        list_of_tp_all                      = []
        list_of_tn_all                      = []
        list_of_fp_all                      = []
        list_of_fn_all                      = []
        list_of_y_true_all_global           = [] 
        list_of_y_predict_all_global        = []

        list_of_count_all                   = []
        list_of_accuracy                    = []
        list_of_precision                   = []
        list_of_recall                      = []
        list_of_f1                          = []
        list_of_specificity                 = []
        list_of_tp_all_check                = []
        list_of_tn_all_check                = []
        list_of_fp_all_check                = []
        list_of_fn_all_check                = []
        list_of_train_and_val               = []
        list_of_test                        = []

        for idx_test in range( num_of_combine ):

            # update the group index 
            # idx_group_use = idx_test

            # initialize the model 
            # get the training dataset
            # get the testing dataset 
            # put them in the global fields
            # checking: 
            idx_group_tmp        = idx_test
            self.init_os_inference( idx_group = idx_group_tmp )
            
            # testing dataset
            data_test_local      = self._data_npy_test
            label_test_local     = self._label_npy_test 

            # count the numebr of points
            num_of_train_and_val = self._data_use_for_train_and_val.shape[0]
            num_of_test          = self._data_npy_test.shape[0]

            # inference     
            # inference of the testing dataset only 
            # self._mean_data_tmp                    = mean_data_tmp
            # self._std_data_tmp                     = std_data_tmp
            # this is coming from the training and validation dataset 
            data_arry_predicted_output = self._unit_mlp_infer_class.infer_npy_dataset( database_npy = data_test_local, 
                                                                                       mean_spectra = self._mean_data_tmp, 
                                                                                       std_spectra  = self._std_data_tmp, 
                                                                                       loss_mode    = "NLLLoss")

            # statistics
            # TODO: implement the functions
            res_acc_sum, count_tp, count_tn, count_fp, count_fn, count_all, list_of_y_true, list_of_y_predict = statistics_from_inference(res_inference = data_arry_predicted_output, 
                                                                                                                                          res_label     = label_test_local)
            
            y_predict_flip  = flipping_label( data_arry_predicted_output )
            y_true_flip     = flipping_label( label_test_local )

            # statistics
            tn, fp, fn, tp      = confusion_matrix( y_true_flip, y_predict_flip ).ravel()

            accuracy        = accuracy_score( y_true_flip, y_predict_flip )
            precision       = precision_score( y_true_flip, y_predict_flip )
            recall          = recall_score( y_true_flip, y_predict_flip)
            f1              = f1_score( y_true_flip, y_predict_flip)
            specificity     = tn / (tn + fp)

            # checking  
            print("data_arry_predicted_output = ", data_arry_predicted_output)
            print("label_test_local = ", label_test_local)
            print("y_predict_flip = ", y_predict_flip)
            print("y_true_flip = ", y_true_flip)
            print("tp = ", tp)
            print("count_tp = ", count_tp)
            print("tn = ", tn)
            print("count_tn = ", count_tn)
            print("fp = ", fp)
            print("count_fp = ", count_fp)
            print("fn = ", fn)
            print("count_fn = ", count_fn)

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

            list_of_tp_all_check.append( tp )
            list_of_tn_all_check.append( tn )
            list_of_fp_all_check.append( fp )
            list_of_fn_all_check.append( fn )
            list_of_accuracy.append( round( accuracy, 2) )
            list_of_precision.append( round( precision, 2) )
            list_of_recall.append( round( recall, 2) )
            list_of_f1.append( round( f1, 2) )
            list_of_specificity.append( round( specificity, 2) )

            # collect the number of data 
            list_of_train_and_val.append(num_of_train_and_val)
            list_of_test.append(num_of_test)

            print("res_acc_sum = ", np.around( res_acc_sum, 2 ) )
            print("count_tp = ", np.around( count_tp, 2 ) )
            print("count_tn = ", np.around( count_tn, 2 ) )
            print("count_fp = ", np.around( count_fp, 2 ) )
            print("count_fn = ", np.around( count_fn, 2 ) )
            print("accuracy = ", np.around( accuracy, 2 ) )
            print("precision = ", np.around( precision, 2 ) )
            print("recall = ", np.around( recall, 2) )
            print("f1 = ", np.around( f1, 2 ) )
            print("specificity = ", np.around( specificity, 2) )

        # summary
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
        
        # np.save( path_statistics_save + "res_train_count.npy", list_of_train_and_val )
        # np.save( path_statistics_save + "res_test_count.npy", list_of_test )
        # np.save( path_statistics_save + "res_tp_count.npy", list_of_tp_all_check )
        # np.save( path_statistics_save + "res_tn_count.npy", list_of_tn_all_check )
        # np.save( path_statistics_save + "res_fp_count.npy", list_of_fp_all_check )
        # np.save( path_statistics_save + "res_fn_count.npy", list_of_fn_all_check )
        # np.save( path_statistics_save + "res_accuracy.npy", list_of_accuracy )
        # np.save( path_statistics_save + "res_precision.npy", list_of_precision )
        # np.save( path_statistics_save + "res_recall.npy", list_of_recall )
        # np.save( path_statistics_save + "res_f1.npy", list_of_f1 )
        # np.save( path_statistics_save + "res_specificity.npy", list_of_specificity )

        # latex table oder == 1) accuracy + 2) precision + 3) recall + 4) F1-score + 5) specificity
        data_pd = [ np.transpose(list_of_accuracy), np.transpose(list_of_precision), np.transpose(list_of_recall), np.transpose(list_of_f1), np.transpose(list_of_specificity) ]
        df = pd.DataFrame( data_pd )
        print("df = ", df)

    def statistics_summary_official(self): 

        # size_of_os       = 15
        # size_of_sts      = 15

        """load the models using the relative paths"""
        path_statistics_os_model  = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/"
        path_statistics_sts_model = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_sts/"  

        # group_idx_os     = np.linspace(0, size_of_os - 1, size_of_os ).astype(np.uint8)
        # TODO: check the group index for the os tumor
        list_of_group_idx_combine_os = [ [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 4, 5], [1, 2, 4, 6], [1, 2, 5, 6], [1, 3, 4, 5], [1, 3, 4, 6], [1, 3, 5, 6], [1, 4, 5, 6], [2, 3, 4, 5], [2, 3, 4, 6], [2, 3, 5, 6], [2, 4, 5, 6], [3, 4, 5, 6] ]
        
        # checking: 
        # print(" list_of_group_idx_combine_os length = ", len( list_of_group_idx_combine_os ) )
        # exit()

        list_of_group_idx_var_os     = []
        for idx_os in range( len(list_of_group_idx_combine_os)): 
            var_idx         = list_of_group_idx_combine_os[idx_os]
            var_tmp         = "OS-" + str(var_idx[0]) + " + OS-" + str(var_idx[1]) + " + OS-" + str(var_idx[2]) + " + OS-" + str(var_idx[3]) + " (idx-" + str(idx_os + 1) + ")"
            list_of_group_idx_var_os.append( var_tmp )

        train_count_os   = np.load( path_statistics_os_model + "res_train_count.npy" )
        test_count_os    = np.load( path_statistics_os_model + "res_test_count.npy" ) 
        tp_count_os      = np.around( np.load( path_statistics_os_model + "res_tp_count.npy" ), decimals=2 )
        tn_count_os      = np.load( path_statistics_os_model + "res_tn_count.npy" ) 
        fp_count_os      = np.load( path_statistics_os_model + "res_fp_count.npy" ) 
        fn_count_os      = np.load( path_statistics_os_model + "res_fn_count.npy" ) 
        accuracy_os      = np.load( path_statistics_os_model + "res_accuracy.npy" )
        precision_os     = np.load( path_statistics_os_model + "res_precision.npy" )
        recall_os        = np.load( path_statistics_os_model + "res_recall.npy" )
        f1_os            = np.load( path_statistics_os_model + "res_f1.npy" )
        specificity_os   = np.load( path_statistics_os_model + "res_specificity.npy" )

        # group_idx_sts     = np.linspace(0, size_of_sts - 1, size_of_sts ).astype(np.uint8)
        # TODO: check the group index for the sts tumor
        list_of_group_idx_combine_sts = [ [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 4, 5], [1, 2, 4, 6], [1, 2, 5, 6], [1, 3, 4, 5], [1, 3, 4, 6], [1, 3, 5, 6], [1, 4, 5, 6], [2, 3, 4, 5], [2, 3, 4, 6], [2, 3, 5, 6], [2, 4, 5, 6], [3, 4, 5, 6] ]

        # checking: 
        # print(" list_of_group_idx_combine_sts length = ", len( list_of_group_idx_combine_sts ) )
        # exit()

        list_of_group_idx_var_sts     = []
        for idx_sts in range( len(list_of_group_idx_combine_sts)): 
            var_idx         = list_of_group_idx_combine_sts[idx_sts]
            var_tmp         = "STS-" + str(var_idx[0]) + " + STS-" + str(var_idx[1]) + " + STS-" + str(var_idx[2]) + " + STS-" + str(var_idx[3]) + " (idx-" + str(idx_sts + 1) + ")"
            list_of_group_idx_var_sts.append( var_tmp ) 

        train_count_sts   = np.load( path_statistics_sts_model + "res_train_count.npy" ) 
        test_count_sts    = np.load( path_statistics_sts_model + "res_test_count.npy" ) 
        tp_count_sts      = np.load( path_statistics_sts_model + "res_tp_count.npy" ) 
        tn_count_sts      = np.load( path_statistics_sts_model + "res_tn_count.npy" ) 
        fp_count_sts      = np.load( path_statistics_sts_model + "res_fp_count.npy" ) 
        fn_count_sts      = np.load( path_statistics_sts_model + "res_fn_count.npy" ) 
        accuracy_sts      = np.load( path_statistics_sts_model + "res_accuracy.npy" )
        precision_sts     = np.load( path_statistics_sts_model + "res_precision.npy" )
        recall_sts        = np.load( path_statistics_sts_model + "res_recall.npy" )
        f1_sts            = np.load( path_statistics_sts_model + "res_f1.npy" )
        specificity_sts   = np.load( path_statistics_sts_model + "res_specificity.npy" )

        # os-model 
        df_os                     = pd.DataFrame()
        df_os['Group index']      = list_of_group_idx_var_os
        df_os['train + val']      = train_count_os
        df_os['testing']          = test_count_os
        df_os['TP']               = tp_count_os
        df_os['TN']               = tn_count_os
        df_os['FP']               = fp_count_os
        df_os['FN']               = fn_count_os
        df_os['Accuracy']         = accuracy_os
        df_os['Accuracy']         = df_os['Accuracy'].apply(lambda x: f'{x:.2f}')
        df_os['Precision']        = precision_os
        df_os['Precision']        = df_os['Precision'].apply(lambda x: f'{x:.2f}')
        df_os['Recall']           = recall_os
        df_os['Recall']           = df_os['Recall'].apply(lambda x: f'{x:.2f}')
        df_os['F1-score']         = f1_os
        df_os['F1-score']         = df_os['F1-score'].apply(lambda x: f'{x:.2f}')
        df_os['Specificity']      = specificity_os
        df_os['Specificity']      = df_os['Specificity'].apply(lambda x: f'{x:.2f}')
        latex_os = df_os.to_latex(index=False)
        print("latex_os = ", latex_os)

        # sts-model
        df_sts                     = pd.DataFrame()
        df_sts['Group index']      = list_of_group_idx_var_sts
        df_sts['train + val']      = train_count_sts
        df_sts['testing']          = test_count_sts
        df_sts['TP']               = tp_count_sts
        df_sts['TN']               = tn_count_sts
        df_sts['FP']               = fp_count_sts
        df_sts['FN']               = fn_count_sts
        df_sts['Accuracy']         = accuracy_sts
        df_sts['Accuracy']         = df_sts['Accuracy'].apply(lambda x: f'{x:.2f}')
        df_sts['Precision']        = precision_sts
        df_sts['Precision']        = df_sts['Precision'].apply(lambda x: f'{x:.2f}')
        df_sts['Recall']           = recall_sts
        df_sts['Recall']           = df_sts['Recall'].apply(lambda x: f'{x:.2f}')
        df_sts['F1-score']         = f1_sts
        df_sts['F1-score']         = df_sts['F1-score'].apply(lambda x: f'{x:.2f}')
        df_sts['Specificity']      = specificity_sts
        df_sts['Specificity']      = df_sts['Specificity'].apply(lambda x: f'{x:.2f}')
        latex_sts = df_sts.to_latex(index=False)
        print("latex_sts = ", latex_sts)

def get_permutations_no_repeat(sequence):
  """
  Generates all permutations of a sequence without repeating elements.

  Args:
    sequence: The input sequence (e.g., string, list, tuple).

  Returns:
    A list of tuples, where each tuple is a permutation of the input sequence.
  """
  return list(itertools.permutations(sequence))

def flipping_label( y_ary ):

    y_ary_flip = []

    for item in y_ary: 

        if item == 0: 
            y_ary_flip.append( 1 )
        else: 
            y_ary_flip.append( 0 )

    return y_ary_flip

if __name__ == "__main__":

    # utility class 
    utility_class = mlp_tumor_degree_analysis()

    # """os tumor model"""
    utility_class.os_mlp_train()
    utility_class.os_mlp_test()

    # """sts tumor model"""
    utility_class.sts_mlp_train()
    utility_class.sts_mlp_test()

    # """post-analysis for the whole models = os + sts"""
    utility_class.statistics_summary_official()