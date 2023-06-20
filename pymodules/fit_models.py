# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:33:21 2022

This script fits H[S]MM models in 1D (r), 2D (LF,HF), and 6D (canonical bands).
The fit models are saved to a model directory using pickle dump.

@author: dweiss38
"""

import ssm
import os
import pickle

from ssm.plots import gradient_cmap
from ssm.util import find_permutation

import seaborn as sns
import autograd.numpy as np
import dim_red_funcs as dr
import dim_red_utils as util


sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

def get_fourier_amplitude(spec, f, band,logtransform=True):
    if logtransform:
        return np.log(np.mean(np.abs(spec[np.squeeze(np.logical_and(f>band[0],f<band[1])),:]),0))
    else:
        return np.mean(np.abs(spec[np.squeeze(np.logical_and(f>band[0],f<band[1])),:]),0)

def get_fourier_amplitudes(spec, f, bands):
    data = []
    for band in bands:
        data.append(get_fourier_amplitude(spec, f, band))
    return np.array(data)

#%%


recordings = ['AP097_3', 'AP098_2', 'AP103_1', 'AP104_2', 'AP101_4']
#recordings = ['AP098_2']

canonical_bands = {'delta': [1,3],
                   'theta': [3,10],
                   'alpha': [10,15],
                   'beta': [15,30],
                   'low-gamma': [30,70],
                   'high-gamma': [70,90]}

fs = 500
Fs_spect = 80/2000 # Sampling rate of the spectrogram

experiment = '4b-bp-03200-fs500-win512ms-step40ms'
data_dir = 'Y:\stanley\Data\CorticalStateData\S1_Amplitude_Spectrums\%s' % experiment
model_dir = '../model/%s' % experiment

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for recording in recordings:
    
    split='train'
    suffix = None
    suffix2 = None
    f,c_train,amplitudes_train,t_train = util.load_recording_data(recording, \
                                                                  data_dir, \
                                                                  split=split, \
                                                                  suffix=suffix)
    
    amplitudes_standardized_train, scaler = dr.fit_scale_data(amplitudes_train);
    
    c_train[c_train == 2] = 0

    ### PCA ###
    # fitting features with PCA
    feature_method_name = 'PCA';
    n_components = 2;
    pca_features,pca_projections,pca = \
        dr.fit_PCA(amplitudes_standardized_train,n_components);
    
    x_train = \
        dr.feature_projection(amplitudes_standardized_train,pca_features);
    
    # Minimum and maximum sub-states in each "super" state
    transition_kwargs = {'r_min': 11,
                         'r_max': 11}
    
    # Set the parameters of the HMM
    T = 5000    # number of time bins
    K = 2       # number of discrete states
    #D = np.shape(spec)[0]       # number of observed dimensions
    D = 2
    
    # Fit an HSMM
    N_em_iters = 100
    
    print("Fitting 2D Gaussian HSMM with EM")
    hsmm = ssm.HSMM(K, D, observations="gaussian",transition_kwargs=transition_kwargs)
    hsmm_em_lls = hsmm.fit(x_train, method="em",num_iters=N_em_iters)
    
    
    # Plot the true and inferred states
    hsmm.permute(find_permutation(c_train, hsmm.most_likely_states(x_train)))
    
    
    print("Writing models to files")
    if suffix2 is None:
        filename_hsmm_pca = '%s\\%s_S1_HSMM2D_pca.pkl' % (model_dir,recording)
        
    else:
        filename_hsmm_pca = '%s\\%s_S1_HSMM2D_pca_%s.pkl' % (model_dir,recording,suffix2)
        
        
    with open(filename_hsmm_pca, 'wb') as pkl:
        pickle.dump([hsmm,pca_features,scaler], pkl)
        
        
        
        
        
        
        
        
        