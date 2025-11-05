
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import random  

# pytorch related studies
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append("./utility/")
import ml_inference_edit_org

random.seed(42)
np.random.seed(42)
torch.manual_seed(0)

class test_internal():

    def __init__(self):
        
        # inference class
        self._unit_mlp_infer_class = ml_inference_edit_org.mlp_inference_unit_test()

    def inference_npy_dataset(self): 
        """prediction for a single .npy datasets"""

        # init: mlp-training model 
        self._unit_mlp_infer_class.init_build_mlp_train_val_model() 

        # init: mlp-inference model
        idx_group                                       = 0
        path_mlp_model_unique                           = "./exp_mice_resection/mouse_mlp_study/result_mlp_cross_validation_six_to_two_os/"
        para_inference_input                            = {}
        para_inference_input["model_name"]              = "tumormap"
        para_inference_input["loss_mode"]               = "NLLLoss"
        para_inference_input["model_dict"]              = path_mlp_model_unique + "pth_res/" 
        para_inference_input["log_instance"]            = "group_idx_" + str( idx_group )
        para_inference_input["device_infer"]            = 'cpu'
        self._unit_mlp_infer_class.init_build_mlp_inference_model( para_inference_input = para_inference_input )

        # training and validation 
        self._unit_mlp_infer_class._path_global_dataset = path_mlp_model_unique + "dataset/"
        path_data_use_for_training                      = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_train_and_val.npy"
        data_npy_train_and_val                          = np.load( path_data_use_for_training )
        self._data_use_for_train_and_val                = data_npy_train_and_val
        mean_data_tmp, std_data_tmp                     = self._unit_mlp_infer_class.train_mean_and_std_from_single_dataset( data_npy_input = data_npy_train_and_val )
        self._mean_data_tmp                             = mean_data_tmp
        self._std_data_tmp                              = std_data_tmp

        # load the testing dataset 
        path_data_use_for_testing                       = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_data_test.npy"
        path_label_use_for_testing                      = self._unit_mlp_infer_class._path_global_dataset + para_inference_input["model_name"] + "_" +  para_inference_input["log_instance"] + "_label_test.npy"
        self._data_npy_test                             = np.load( path_data_use_for_testing )
        self._label_npy_test                            = np.load( path_label_use_for_testing )

        # inference 
        data_test_local      = self._data_npy_test
        label_test_local     = self._label_npy_test 

        # inference     
        data_arry_predicted_output = self._unit_mlp_infer_class.infer_npy_dataset( database_npy = data_test_local, 
                                                                                    mean_spectra = self._mean_data_tmp, 
                                                                                    std_spectra  = self._std_data_tmp, 
                                                                                    loss_mode    = "NLLLoss")

        print("the predictions of the npy dataset is ", data_arry_predicted_output )
        print("The labels are ", label_test_local)

if __name__ == "__main__":

    # define the class
    unit_test_class = test_internal()

    """inference: folder or npy array"""
    unit_test_class.inference_npy_dataset()
