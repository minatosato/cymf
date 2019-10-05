# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn.metrics import roc_auc_score

def auc(model, X):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    scores = model.W.dot(model.H.T)
    return np.array([roc_auc_score(user_ratings, score) if len(set(user_ratings)) == 2 else 0 for score, user_ratings in zip(scores, X)])

def precision_at_k(model, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    scores = model.W.dot(model.H.T)
    return np.array([user_ratings[indices].mean() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)])

def recall_at_k(model, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    scores = model.W.dot(model.H.T)
    bunbo = X.sum(axis=1)
    bunbo[bunbo == 0] = 1
    return np.array([user_ratings[indices].sum() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)]) / bunbo

def dcg_at_k(model, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    scores = model.W.dot(model.H.T)
    bunbo = np.array([((np.ones(len(indices)) / np.log2(np.arange(2, k+2)))).sum() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)])
    return np.array([(user_ratings[indices] / np.log2(np.arange(2, k+2))).sum() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)]) / bunbo


def precision(scores, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return np.array([user_ratings[indices].mean() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)])

def recall(scores, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    bunbo = X.sum(axis=1)
    bunbo[bunbo == 0] = 1
    return np.array([user_ratings[indices].sum() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)]) / bunbo

def dcg(scores, X, k=10):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    bunbo = np.array([(np.ones(k) / np.log2(np.arange(2, k+2))).sum() for i in range(len(X))])
    return np.array([(user_ratings[indices] / np.log2(np.arange(2, k+2))).sum() for indices, user_ratings in zip(scores.argsort(axis=1)[:,::-1][:,:k], X)]) / bunbo