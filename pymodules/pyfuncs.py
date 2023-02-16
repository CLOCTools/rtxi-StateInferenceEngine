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

LOG_EPS = 1e-16
DIV_EPS = 1e-16

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
    obs = dr.feature_projection(dr.scale_data(np.array(data),scaler),feats)
    # model.filter is erroring on assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8 
    # (line 94 of messages). For now do viterbi but we should switch back to 
    # filter later
    
    #Ps = model.transitions.transition_matrices(obs, input=None, mask=None, tag=None)
    #Ps = np.exp(model.transitions.log_Ps)
    
    #log_likes = model.observations.log_likelihoods(obs, input=None, mask=None, tag=None)
    #mus = model.observations.mus
    #Sigmas = model.observations.Sigmas
    #pi0 = model.init_state_distn.initial_state_distn
    #P = model.transitions.transition_matrix
    #Ps = transition_matrices(P, obs)
    log_likes = log_likelihoods(mus, Sigmas, obs)
    #m = model.state_map
    #return [1,1,1]
    return m[viterbi(replicate(pi0, m), Ps, replicate(log_likes, m))]

def initialize(model, data):
    global pi0 
    global Ps
    global mus
    global Sigmas
    global m

    pi0 = model.init_state_distn.initial_state_distn
    P = model.transitions.transition_matrix
    Ps = transition_matrices(P, np.array(data))
    mus = model.observations.mus
    Sigmas = model.observations.Sigmas
    m = model.state_map

        
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


def viterbi(pi0, Ps, ll):
    """
    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)
    if not hetero:
        print(Ps.shape)
        assert Ps.shape[0] == 1

    # Pass max-sum messages backward
    scores = np.zeros((T, K))
    args = np.zeros((T, K))
    for t in range(T-2,-1,-1):
        vals = np.log(Ps[t * hetero] + LOG_EPS) + scores[t+1] + ll[t+1]
        for k in range(K):
            args[t+1, k] = np.argmax(vals[k])
            scores[t, k] = np.max(vals[k])

    # Now maximize forwards
    z = np.zeros(T)
    z[0] = (scores[0] + np.log(pi0 + LOG_EPS) + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, int(z[t-1])]

    return z.astype(int)


## NOTE: THESE NEED self.log_Ps from model.transitions
## transition class is ssm.transitions.StationaryTransitions
def transition_matrices(transition_matrix, data):
    return np.exp(log_transition_matrices(transition_matrix, data))

def log_transition_matrices(transition_matrix, data):
    T = data.shape[0]
    return np.tile(np.log(transition_matrix)[None, :, :], (T-1, 1, 1))

## NOTE: THESE NEED self.mus and self.Sigmas from model.observations
## observation class is ssm.observations.GaussianObservations
import ssm.stats as stats
def log_likelihoods(mus, Sigmas, data, mask=None):
    if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
        raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                        "does not work with autograd because it writes to an array. "
                        "Use DiagonalGaussian instead if you need to support missing data.")

    # stats.multivariate_normal_logpdf supports broadcasting, but we get
    # significant performance benefit if we call it with (TxD), (D,), and (D,D)
    # arrays as inputs
    return np.column_stack([stats.multivariate_normal_logpdf(data, mu, Sigma)
                           for mu, Sigma in zip(mus, Sigmas)])



def most_likely_states(m, pi0, Ps, log_likes, state_map, data):
    # m = self.state_map
    # pi0 = self.init_state_distn.initial_state_distn
    # Ps = self.transitions.transition_matrices(data, input, mask, tag)
    # log_likes = self.observations.log_likelihoods(data, input, mask, tag)
    z_star = viterbi(replicate(pi0, m), Ps, replicate(log_likes, m))
    return state_map[z_star]

# def expected_states(self, data, input=None, mask=None, tag=None):
#     m = self.state_map
#     pi0 = self.init_state_distn.initial_state_distn
#     Ps = self.transitions.transition_matrices(data, input, mask, tag)
#     log_likes = self.observations.log_likelihoods(data, input, mask, tag)
#     Ez, Ezzp1, normalizer = hmm_expected_states(replicate(pi0, m), Ps, replicate(log_likes, m))

#     # Collapse the expected states
#     Ez = collapse(Ez, m)
#     Ezzp1 = collapse(collapse(Ezzp1, m, axis=2), m, axis=1)
#     return Ez, Ezzp1, normalizer

# @ensure_args_not_none
# def filter(self, data, input=None, mask=None, tag=None):
#     m = self.state_map
#     pi0 = self.init_state_distn.initial_state_distn
#     Ps = self.transitions.transition_matrices(data, input, mask, tag)
#     log_likes = self.observations.log_likelihoods(data, input, mask, tag)
#     pzp1 = hmm_filter(replicate(pi0, m), Ps, replicate(log_likes, m))
#     return collapse(pzp1, m)

def replicate(x, state_map, axis=-1):
    """
    Replicate an array of shape (..., K) according to the given state map
    to get an array of shape (..., R) where R is the total number of states.

    Parameters
    ----------
    x : array_like, shape (..., K)
        The array to be replicated.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R)
    """
    assert state_map.ndim == 1
    assert np.all(state_map >= 0) and np.all(state_map < x.shape[-1])
    return np.take(x, state_map, axis=axis)