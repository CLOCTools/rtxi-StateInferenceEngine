# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:36:47 2022

@author: aborsa3
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis

def knn_accuracy(X_train,c_train,X_test,c_test,n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors);
    knn.fit(X_train,c_train)
    c_pred = knn.predict(X_test);
    return metrics.balanced_accuracy_score(c_test,c_pred)

def silhouette_score(X,c,metric='euclidean'):
    return metrics.silhouette_score(X,c,metric=metric)

def qda_accuracy(X_train,c_train,X_test,c_test):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train,c_train)
    
    c_pred = qda.predict(X_test)
    return metrics.balanced_accuracy_score(c_test,c_pred)

def lda_accuracy(X_train,c_train,X_test,c_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train,c_train)
    
    c_pred = lda.predict(X_test)
    return metrics.balanced_accuracy_score(c_test,c_pred)

def normalize_features(features):
    # features should have size dimension of data x number of features
    return features / np.linalg.norm(features,ord=1,axis=0)

def feature_projection(X,features,norm=True): 
    # X should have size num_samples x dimension of data
    # features should have size dimension of data x number of features
    if norm:
        return X @ normalize_features(features)
    else:
        return X @ features

def fit_scale_data(X):
    scaler = StandardScaler();
    return scaler.fit_transform(X), scaler

def scale_data(X,scaler):
    return scaler.transform(X)

def split_data(X,c,test_size=0.3,shuffle=False):
    X_train, X_test, c_train, c_test = train_test_split(X,c,test_size=test_size,shuffle=shuffle)
    return X_train, X_test, c_train, c_test

def fit_PCA(X,n_components): # X should have size num_samples x dimension of data
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    features = pca.components_.T;
    norm_features = normalize_features(features);
    projections = feature_projection(X,features);
    return norm_features, projections, pca

def fit_SPCA(X,n_components,alpha):
    spca = SparsePCA(n_components=n_components,alpha=alpha)
    spca.fit(X)
    
    features = spca.components_.T;
    norm_features = normalize_features(features);
    projections = feature_projection(X,features)
    return norm_features, projections, spca

def fit_NCA(X,c,n_components,init='pca'):
    nca = NeighborhoodComponentsAnalysis(n_components=n_components,init=init,\
                                         max_iter=500,verbose=1)
    nca.fit(X,c)
    
    features = nca.components_.T;
    norm_features = normalize_features(features);
    projections = feature_projection(X,features)
    return norm_features, projections, nca

def make_LFHF_features(f_vec,lf_low,lf_high,hf_low,hf_high):
    features = np.array([np.all([f_vec>lf_low,f_vec<lf_high],axis=0),\
                         np.all([f_vec>hf_low,f_vec<hf_high],axis=0)]).T
    norm_features = features / np.sum(features, axis=0)
    return norm_features