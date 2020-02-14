# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++

import cython
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cimport cython
from libcpp cimport bool

from .metrics import dcg_at_k
from .metrics import recall_at_k
from .metrics import average_precision_at_k

from .metrics import dcg_at_k_with_ips
from .metrics import recall_at_k_with_ips
from .metrics import average_precision_at_k_with_ips



class Evaluator(object):
    def __init__(self, X,
                       X_train = None,
                       list metrics = ["DCG", "Recall", "MAP"],
                       k = 5,
                       int num_negatives = 100,
                       bool unbiased = False):
        self.X = X
        self.X_train = X_train

        if not isinstance(self.X, np.ndarray):
            raise TypeError("X must be a type of numpy.ndarray.")

        if self.X_train is not None and not isinstance(self.X_train, np.ndarray):
            raise TypeError("X_train must be a type of numpy.ndarray.")

        self.propensity_scores = np.maximum(X.mean(axis=0), 1e-3)
        self.metrics = metrics
        self.k = k
        self.num_negatives = num_negatives
        self.unbiased = unbiased

    def evaluate(self, np.ndarray[double, ndim=2] scores):
        cdef dict buff = {}
        cdef str metric
        cdef np.ndarray[np.int_t, ndim=1] positives
        cdef np.ndarray[np.int_t, ndim=1] negatives
        cdef int user, U
        cdef int k

        U = self.X.shape[0]

        if type(self.k) == int:
            self.k = [self.k]

        for k in self.k:
            for metric in self.metrics:
                buff[f"{metric)}@{k}"] = np.zeros(U)
        
        for user in range(U):
            positives = self.X[user].nonzero()[0]
            negatives = np.random.permutation(
                (self.X[user]==0.).nonzero()[0] if self.X_train is None else ((self.X[user]-self.X_train[user])==0.).nonzero()[0]
            )[:self.num_negatives]

            items = np.r_[positives, negatives]
            ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)

            for k in self.k:
                for metric in self.metrics:
                    if self.unbiased:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k_with_ips(ratings, scores[user, items], k, self.propensity_scores)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k_with_ips(ratings, scores[user, items], k, self.propensity_scores)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k_with_ips(ratings, scores[user, items], k, self.propensity_scores)
                    else:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k(ratings, scores[user, items], k)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k(ratings, scores[user, items], k)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k(ratings, scores[user, items], k)
        
        for k in self.k:
            for metric in self.metrics:
                buff[f"{metric)}@{k}"] = buff[f"{metric)}@{k}"].mean()

        return buff

