

import numpy as np 
import cv2
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import sklearn 
import glob 
import os
import matplotlib.patches as mpatches

class feature_single_tumorid_analysis():

    def __init__(self, main_repo_input = "/", repo_type = "os_tumor_1/", data_index_name = []): 
        """local folder path w.r.t the main repository
        1. self.main_repo: date of the database.
        2. self.mice_repo: types of the mice tissues
        3. self.fixed_point_scan_repo
        4. self.line_scan_repo
        5. self.planar_scan_repo
        6. self.surface_scan_repo
        7. self.oct_scan_repo
        8. self.data_index_name
        """
        
        self.main_repo = main_repo_input
        self.mice_repo = self.main_repo + repo_type 
        self.fixed_point_scan_repo = self.mice_repo + "fixed_point_scan/"
        self.line_scan_repo = self.mice_repo + "line_scan/"
        self.planar_scan_repo = self.mice_repo + "planar_scan/"
        self.surface_scan_repo = self.mice_repo + "surface_raster/"
        self.oct_scan_repo = self.mice_repo + "oct/"
        self.data_index_name = data_index_name      

    # def initialization_of_file_folder(self): 
        a = 1

    def normalize_min_and_max(self, spectra_input = [],  wavelength_norm = [] ): 
        """min-and-max normalization""" 
        spectra_output = spectra_input / wavelength_norm

        return spectra_output

    def preprocessing_new(self, para_dict = {}): 
        """TODO: use the new signal preprocessing method
        1. input: raw signal
        2. cut-off between 450 to 750 (thorlabs)
        3. "each" signal divide the max(signal)
        """

        # parameter settings
        idx_low_cut                     = para_dict["idx_low_cut"] 
        idx_high_cut                    = para_dict["idx_high_cut"] 
        flag_normalization              = para_dict["flag_normalization"]
        flag_smoothing_operator         = para_dict["flag_smoothing_operator"]
        wavelength                      = para_dict["wavelength"]
        spectra                         = para_dict["spectra"]
        flag_cutoff                     = para_dict["flag_cutoff"]
        flag_normalization              = para_dict["flag_normalization"]

        # step-1: range threshold -> keep spectra between 450-750 nm
        if flag_cutoff == "true": 
            idx_range = np.where( (wavelength > idx_low_cut) & (wavelength < idx_high_cut) )[0]
            wavelength = wavelength[idx_range]
            spectra = spectra[idx_range]

        # plt.plot(wavelength, spectra)
        # plt.xlim([para_dict["idx_low_cut"], para_dict["idx_high_cut"]])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("wavelength (cut-off)", fontsize = 20)
        # plt.ylabel("normalized intensity", fontsize = 20)
        # plt.title("single-data-test (after cut off (raw))", fontsize = 20)
        # plt.tick_params(axis='both', labelsize=20)
        # plt.grid()
        # plt.show() 
        # # exit()
    
        # step-2: normalization 
        # min-max normalization
        # min (signal) can be close to zero in this process.
        # TODO: check the min(cut-off signal)
        if flag_normalization == "true":
            
            val_normal_factor_wavelengh = np.max( spectra )
            
            # print("val_normal_factor_wavelengh = ", val_normal_factor_wavelengh)
            # print("spectra = ", spectra)
            # input("check")
            spectra = self.normalize_min_and_max( spectra_input = spectra, wavelength_norm = val_normal_factor_wavelengh )
            # spectra = self.normalize( wavelength_input = wavelength, spectra_input= spectra, wavelength_norm = val_normal_factor_wavelengh )

        # plt.plot(wavelength, spectra)
        # plt.xlim([para_dict["idx_low_cut"], para_dict["idx_high_cut"]])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("wavelength (cut-off)", fontsize = 20)
        # plt.ylabel("normalized intensity", fontsize = 20)
        # plt.title("single-data-test (normalization)", fontsize = 20)
        # plt.tick_params(axis='both', labelsize=20)
        # plt.grid()
        # plt.show() 
        # # exit()

        # step-3: smoothing 
        if flag_smoothing_operator == "savgol":
            spectra_savgol = savgol_filter( spectra, window_length = 10, polyorder = 3 )
            spectra = spectra_savgol
            
        elif flag_smoothing_operator == "convolution":
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            spectra_conv = np.convolve(spectra, kernel, 'same')
            spectra = spectra_conv

        # plt.plot(wavelength, spectra)
        # plt.xlim([para_dict["idx_low_cut"], para_dict["idx_high_cut"]])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("wavelength (cut-off)", fontsize = 20)
        # plt.ylabel("normalized intensity", fontsize = 20)
        # plt.title("single-data-test (after filter)", fontsize = 20)
        # plt.tick_params(axis='both', labelsize=20)
        # plt.grid()
        # plt.show() 
        # # exit()

        return wavelength, spectra
