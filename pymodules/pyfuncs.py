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
import numba

from numba.pycc import CC

LOG_EPS = 1e-16
DIV_EPS = 1e-16

def get_model(recording, experiment, model_dir='./model'):
    file = '%s/%s/%s_S1_HSMM2D_pca.pkl' % (model_dir, experiment, recording)
    with open(file, 'rb') as pkl:
        model, feats, scaler = pickle.load(pkl)
    return model, feats, scaler, model.state_map

def get_data(recording, experiment):
    data_dir = './data/%s' % experiment
    f,c,data,t = util.load_recording_data(recording, data_dir,'test',suffix=None)
    return data

## NOTE: THESE NEED self.log_Ps from model.transitions
## transition class is ssm.transitions.StationaryTransitions
def transition_matrices(transition_matrix, data):
    return np.exp(log_transition_matrices(transition_matrix, data))

def log_transition_matrices(transition_matrix, data):
    T = data.shape[0]
    return np.tile(np.log(transition_matrix)[None, :, :], (T-1, 1, 1))

## NOTE: THESE NEED self.mus and self.Sigmas from model.observations
## observation class is ssm.observations.GaussianObservations
#import ssm.stats as stats
def log_likelihoods(mus, Sigmas, data, mask=None):
    if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
        raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                        "does not work with autograd because it writes to an array. "
                        "Use DiagonalGaussian instead if you need to support missing data.")

    # stats.multivariate_normal_logpdf supports broadcasting, but we get
    # significant performance benefit if we call it with (TxD), (D,), and (D,D)
    # arrays as inputs
    return np.column_stack([multivariate_normal_logpdf(data, mu, Sigma)
                           for mu, Sigma in zip(mus, Sigmas)])

def flatten_to_dim(X, d):
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])

from autograd.scipy.linalg import solve_triangular

#@numba.jit(nopython=True,cache=True)
def batch_mahalanobis(L, x):
    # The most common shapes are x: (T, D) and L : (D, D)
    # Special case that one
    if x.ndim == 2 and L.ndim == 2:
        xs = solve_triangular(L, x.T, lower=True)
        return np.sum(xs**2, axis=0)

    # Flatten the Cholesky into a (-1, D, D) array
    flat_L = flatten_to_dim(L, 2)
    # Invert each of the K arrays and reshape like L
    L_inv = np.reshape(np.array([np.linalg.inv(Li.T) for Li in flat_L]), L.shape)
    # dot with L_inv^T; square and sum.
    xs = np.einsum('...i,...ij->...j', x, L_inv)
    return np.sum(xs**2, axis=-1)


#@numba.jit(nopython=True,cache=True)
def multivariate_normal_logpdf(data, mus, Sigmas, Ls=None):
    
    # Check inputs
    D = data.shape[-1]
    #assert mus.shape[-1] == D
    #assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    #if Ls is not None:
    #    assert Ls.shape[-2] == Ls.shape[-1] == D
    #else:
    Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    lp = -0.5 * batch_mahalanobis(Ls, data - mus)                    # (...,)
    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp

def most_likely_states(m, pi0, Ps, log_likes, state_map, data):
    # m = self.state_map
    # pi0 = self.init_state_distn.initial_state_distn
    # Ps = self.transitions.transition_matrices(data, input, mask, tag)
    # log_likes = self.observations.log_likelihoods(data, input, mask, tag)
    z_star = viterbi(replicate(pi0, m), Ps, replicate(log_likes, m))
    return state_map[z_star]

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
    #return model.filter(obs)

def log_likes(model, feats, scaler, data):
    obs = dr.feature_projection(dr.scale_data(np.array(data),scaler),feats)
    log_like = log_likelihoods(mus, Sigmas, obs)
    T,K = replicate(log_like, m).shape
    return list(replicate(pi0, m).flatten(order='F')), list(Ps.flatten(order='F')), list(replicate(log_like, m).flatten(order='F')), T, K

def initialize(model, data):
    global pi0 
    global Ps
    global mus
    global Sigmas
    global m

    pi0 = model.init_state_distn.initial_state_distn
    Ps = model.transitions.transition_matrix
    #Ps = transition_matrices(P, np.array(data))
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

@numba.jit(nopython=True, cache=True)
def _viterbi(pi0, Ps, ll):
    """
    This is modified from pyhsmm.internals.hmm_states
    by Matthew Johnson.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)

    # Pass max-sum messages backward
    scores = np.zeros((T, K))
    args = np.zeros((T, K))
    for t in range(T-2,-1,-1):
        vals = np.log(Ps + LOG_EPS) + scores[t+1] + ll[t+1]
        for k in range(K):
            args[t+1, k] = np.argmax(vals[k])
            scores[t, k] = np.max(vals[k])

    # Now maximize forwards
    z = np.zeros(T)
    z[0] = (scores[0] + np.log(pi0 + LOG_EPS) + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, int(z[t-1])]

    return z

def viterbi(pi0,Ps,ll):
    return _viterbi(pi0,Ps,ll).astype(int)

#import viterbi_test

#def viterbi(pi0,Ps,ll):
#    return viterbi_test.viterbi(pi0,Ps,ll).astype(int)

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

def compiler():
    cc.compile()