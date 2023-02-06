# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:30:02 2023

predict.py is used to infer a state given a set of measurements, previous states,
and a model.

@author: dweiss38
"""

import pickle
import dim_red_funcs as dr
import dim_red_utils as util
import numpy as np

def get_model(recording, experiment, model_dir='./model'):
    file = '%s/%s_S1_%s.pkl' % (model_dir, recording, experiment)
    with open(file, 'rb') as pkl:
        model, feats, scaler = pickle.load(pkl)
    return model, feats, scaler

def get_data(recording, experiment):
    data_dir = './data/%s' % experiment
    f,c,data,t = util.load_recording_data(recording, data_dir,'test',suffix=None)
    return data

def predict(model, feats, scaler, data):
    # Here we're going to force data to be an array since
    # C++ doesn't have numpy data types
    obs = dr.feature_projection(dr.scale_data(data,scaler),feats)
    # model.filter is erroring on assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8 
    # (line 94 of messages). For now do viterbi but we should switch back to 
    # filter later
    return model.most_likely_states(obs)
        
def infer():
    # Specify model location, to be replaced with command line argument
    recording = 'AP103_1' 
    experiment = '4b-bp-03200-fs500-win512ms-step40ms'
    # I don't like that this is hard-coded
    # Maybe we can make this build dependent
    model, feats, scaler = get_model(recording, experiment)
    
    # Specify data, to be replaced with data stream
    data_dir = './data/%s' % experiment
    f,c,data,t = util.load_recording_data(recording,data_dir,'test', suffix=None)
    return predict(model, feats, scaler, data)