
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap

# pytorch related studies
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ml_deeplearning_edit_org import *
from feature_engineering_edit_org import *
import ml_model_train_edit_org

class mlp_inference_unit_test(): 

    def __init__(self):

        # define the feature class module 
        self.tumorid_analysis           = feature_single_tumorid_analysis()

        # user select the model type
        # self.flag_model           = "os_40_degree"
        self.flag_model                 = "os_90_degree"
        # self.flag_model           = "sts_40_degree"
        # self.flag_model           = "sts_90_degree"

        # path
        self._path_global_check         = "./mouse_mlp_study/data_loss_acc_result/"
        self._path_global_dataset       = "./mouse_mlp_study/data_tumormap_official/"
        self._path_global_save_mlp      = "./mouse_mlp_study/data_mlp_model_official/"
        self._log_dir                   = "./mouse_mlp_study/mlp_train_log_dir/"
    
        # model parameters 
        self._val_batchsize             = 16
        self._val_numepoch              = 150
        self._lr                        = 0.001
        self._idx_use_gpu               = 0
        self._is_use_cpu                = False

        # data mode
        if self.flag_model   == "os_40_degree": 
            self._val_tumor_type   = "OS"
            self._val_log_instance = "os_40_degree"
        elif self.flag_model == "os_90_degree": 
            self._val_tumor_type   = "OS"
            self._val_log_instance = "os_90_degree"
        elif self.flag_model == "sts_40_degree": 
            self._val_tumor_type   = "STS"
            self._val_log_instance = "sts_40_degree"
        elif self.flag_model == "sts_90_degree": 
            self._val_tumor_type   = "STS"
            self._val_log_instance = "sts_90_degree"
        else:
            print("please input the model type")
            exit()

    def init_build_mlp_inference_model(self, para_inference_input = {} ): 
        """
        # # build the model  
        # self.model_name     = "tumor_map"
        # self.loss_mode      = "NLLLoss"
        # self.model_dict     = self._path_global_save_mlp
        # self.log_instance   = self._val_log_instance
        """

        # init
        model_name_infer   = para_inference_input["model_name"]
        loss_mode_infer    = para_inference_input["loss_mode"]
        model_dict_infer   = para_inference_input["model_dict"]
        log_instance_infer = para_inference_input["log_instance"]
        device_infer       = para_inference_input["device_infer"]  # 'cpu'

        # build the model
        if model_name_infer == "tumormap":
            self._mlp_inference_model = SpectralClassification(loss_mode = loss_mode_infer)
        else:
            print("Model name not found")
            sys.exit(1)

        # load the pretrained model
        # evaluation for the loading of the models
        # Dropout layers are deactivated:
        # Batch Normalization layers use running statistics:
        path_model_pth_file = model_dict_infer + model_name_infer + "_" + log_instance_infer + ".pth"
        self._mlp_inference_model.load_state_dict( torch.load( path_model_pth_file, map_location = torch.device( device_infer )) )
        self._mlp_inference_model.eval()

        # checking
        # input("check the model configuration")
        # return mlp_inference_model
        # print("device_infer = ", device_infer)
        # print(self._mlp_inference_model)
        # print("path_model_pth_file = ", path_model_pth_file)
        # exit()

    def init_build_mlp_train_val_model(self): 
        """build the TumorMap model for train-and-validation"""

        parse_train_and_val  =  {
                                    "gpu":              self._idx_use_gpu,
                                    "use_cpu":          self._is_use_cpu,               # True
                                    "batch_size":       self._val_batchsize,            # 16,
                                    "num_epochs":       self._val_numepoch,             # 150,
                                    "lr":               self._lr,
                                    "model_name":       "tumor_map",
                                    "path_save_model":  self._path_global_save_mlp, 
                                    "log_dir":          self._log_dir,
                                    "flag":             "train",
                                    "tumor_type":       self._val_tumor_type,           # "STS" or "OS"
                                    "log_instance":     self._val_log_instance,       # "sts_official_1" or "os_official_1"
                                    "loss_mode":        "NLLLoss"                  # NLLLoss or cross_entropy
                                }

        self._mlp_train_model      = ml_model_train_edit_org.TumorMAP( parse = parse_train_and_val, 
                                                                   path_of_dataset = "./mouse_mlp_study/data_tumormap_official/" ) 
        print("mlp_train_model = ", self._mlp_train_model)
        
        # return mlp_train_model

    def train_mean_and_std_from_single_dataset(self, data_npy_input = []): 

        mean_data_tmp, std_data_tmp = self._mlp_train_model.dataset_normalize( dataset = data_npy_input )
        
        return mean_data_tmp, std_data_tmp

    def infer_npy_data_check(self, para_input = {} ): 

        # init
        # path_test_folder                            = para_input["path_test_folder"] 
        mean_spectra                                = para_input["mean_spectra"]      
        std_spectra                                 = para_input["std_spectra"]      
        loss_mode                                   = para_input["loss_mode"]         
        data_npy_stich_wv_raw                       = para_input["data_npy_stich_wv_raw"]
        data_npy_stich_sp_raw                       = para_input["data_npy_stich_sp_raw"]

        # load the folder 
        num_of_file                                 = np.int64( data_npy_stich_sp_raw.shape[0]  )

        # loop for each file (with the models)
        data_arry_predicted_output                  = []

        for idx_file in range(num_of_file): 

            # load the data
            para_dict = {}
            
            para_dict["wavelength"]                 = data_npy_stich_wv_raw[idx_file,:] 
            para_dict["spectra"]                    = data_npy_stich_sp_raw[idx_file,:] 

            para_dict["flag_cutoff"]                = "true"
            para_dict["flag_normalization"]         = "true"
            para_dict["idx_low_cut"]                = 450 
            para_dict["idx_high_cut"]               = 750
            para_dict["flag_smoothing_operator"]    = "savgol"
            wavelength_post, spectra_post           = self.tumorid_analysis.preprocessing_new(para_dict=para_dict)
            spectra_post                            = spectra_post.reshape(1,-1)        # spectra_post shape =  (1, 1301)

            # # checking 
            # # check-1: 
            # # check-2:
            # print("spectra_post shape = ", spectra_post.shape)
            # exit()
            # # spectra_post_check_1 = spectra_post
            # # spectra_post_check_2 = spectra_post.reshape(1,-1)
            # sp_tumor_model  = spectra_post
            # sp_tumor_model  = np.vstack([sp_tumor_model, spectra_post])
            # sp_tumor_model_rpj = sp_tumor_model[0,:]
            # print("spectra_post shape = ", spectra_post.shape)
            # print("sp_tumor_model shape = ", sp_tumor_model.shape)
            # print("sp_tumor_model_rpj = ", sp_tumor_model_rpj.shape)
            # # plt.plot( wavelength_post, spectra_post, color='r', linewidth = 8 )
            # # plt.plot( wavelength_post, sp_tumor_model_rpj, color='b', linewidth = 3 )
            # # plt.show()
            # exit()

            # normalize the test data with its 
            spectra_post                            = (spectra_post - mean_spectra) / std_spectra
            spectra_post_tensor                     = torch.from_numpy(spectra_post).float()

            # predict the output
            output = self._mlp_inference_model(spectra_post_tensor)

            # prediction 
            if loss_mode == "cross_entropy":
                output = F.softmax(output, dim = 1)
                output = output.detach().numpy()
            elif loss_mode == "NLLLoss":
                output = output
                output = output.detach().numpy() 
            
            # predicted label 
            if output[0][0] > output[0][1]:
                # print("tumor")
                label_predict_tmp = 0
            else:
                # print("healthy")
                label_predict_tmp = 1

            # summarize the output prediction results
            data_arry_predicted_output.append(label_predict_tmp)

        return data_arry_predicted_output

    def infer_npy_dataset(self, database_npy = [], mean_spectra = [], std_spectra = [], loss_mode = []): 
        """input the database as the .npy file
        1. cross-validation
        2. testing verification. 
        """

        # loop for each file (with the models)
        # file information 
        num_of_file                 = database_npy.shape[0]
        data_arry_predicted_output  = []

        for idx_file in range(num_of_file): 

            # load the data
            spectra_post        = database_npy[idx_file,:]
            spectra_post        = spectra_post.reshape(1,-1)

            # # checking
            # print("spectra_post shape = ", spectra_post.shape)
            # print("spectra_post = ", spectra_post.ravel())
            # # num_max_use = spectra_post.shape[0]
            # # print("num_max_use = ", num_max_use)
            # x_check_for_spectra_post = np.linspace( 0, spectra_post.shape[1], spectra_post.shape[1] )
            # # print("x_check_for_spectra_post = ", x_check_for_spectra_post)
            # plt.plot( x_check_for_spectra_post, spectra_post.ravel() )
            # plt.show() 
            # exit()

            # normalize the test data with its 
            spectra_post        = (spectra_post - mean_spectra) / std_spectra
            spectra_post_tensor = torch.from_numpy(spectra_post).float()
            # checking: 
            # spectra_post_tensor shape =  torch.Size([1, 1301])
            # print("spectra_post_tensor shape = ", spectra_post_tensor.shape)
            # exit()

            # predict the output
            # output              = self._mlp_inference_model(spectra_post_tensor)

            # predict the output
            output              = self._mlp_inference_model(spectra_post_tensor)
            
            # # checking 
            # output_check        = output[0].detach().numpy()
            # sum_output_check    = output_check[0] + output_check[1]
            # print("output_check = ", output_check)
            # print("sum_output_check = ", sum_output_check)
            # input("checking")

            if loss_mode == "cross_entropy":
                output = F.softmax(output, dim = 1)
                output = output.detach().numpy()
            elif loss_mode == "NLLLoss":
                output = output
                output = output.detach().numpy() 
            
            # predicted label 
            if output[0][0] > output[0][1]:
                # print("tumor")
                label_predict_tmp = 0
            else:
                # print("healthy")
                label_predict_tmp = 1

            data_arry_predicted_output.append(label_predict_tmp)

        return data_arry_predicted_output
