# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:26:40 2022

@author: aborsa3
"""

import numpy as np
import mat73
import scipy.io as sio
from os.path import join


def load_recording_data(recording_name,data_dir, split=None,suffix=None):
    if split is None:
        raise Exception("Must specify train or test split.")
    if suffix is None:
        filename = recording_name + '_S1_preproc_%s.mat' % split
    else:
        filename = recording_name + '_S1_%s_preproc_%s.mat' % (suffix, split)  
    fullfilename = join(data_dir,filename)
    try:
        data = sio.loadmat(fullfilename)
    except:
        data = mat73.loadmat(fullfilename)
    s = data['s']
    c = np.squeeze(data['c']).astype(np.int32)
    f = np.squeeze(data['f'])
    t = np.squeeze(data['t'])
    
    # consider only frequencies from 1-100 Hz
    indices_relevant_freqs = np.squeeze(np.all([f>1,f<100],axis=0))
    f = f[indices_relevant_freqs]
    s = s[indices_relevant_freqs,:]
    
    # take every 25th data point so that there is 1 second between spectrums
    # s = s[:,0:-1:25];
    # c = c[0:-1:25];
    
    # only consider data points where whisking or quiet (no indeterminate case)
    indices_two_categories = np.squeeze(c != 3)
    c = c[indices_two_categories]
    s = s[:,indices_two_categories]
    
    amplitudes = np.abs(s.T)
    return f, c, amplitudes, t
