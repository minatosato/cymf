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
    def __init__(self, np.ndarray[double, ndim=2] X, list metrics = ["DCG", "Recall", "MAP"], int k = 5, int num_negatives = 100, bool unbiased = False):
        self.X = X
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
        cdef int user

        for metric in self.metrics:
            buff[f"{metric)}@{self.k}"] = np.zeros(self.X.shape[0])
        
        for user in range(self.X.shape[0]):
            positives = self.X[user].nonzero()[0]
            negatives = np.random.permutation((self.X[user]==0.).nonzero()[0])[:self.num_negatives]

            items = np.r_[positives, negatives]
            ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)
            
            for metric in self.metrics:
                if self.unbiased:
                    if metric == "DCG":
                        buff[f"{metric}@{self.k}"][user] = dcg_at_k_with_ips(ratings, scores[user, items], self.k, self.propensity_scores)
                    elif metric == "Recall":
                        buff[f"{metric}@{self.k}"][user] = recall_at_k_with_ips(ratings, scores[user, items], self.k, self.propensity_scores)
                    elif metric == "MAP":
                        buff[f"{metric}@{self.k}"][user] = average_precision_at_k_with_ips(ratings, scores[user, items], self.k, self.propensity_scores)
                else:
                    if metric == "DCG":
                        buff[f"{metric}@{self.k}"][user] = dcg_at_k(ratings, scores[user, items], self.k)
                    elif metric == "Recall":
                        buff[f"{metric}@{self.k}"][user] = recall_at_k(ratings, scores[user, items], self.k)
                    elif metric == "MAP":
                        buff[f"{metric}@{self.k}"][user] = average_precision_at_k(ratings, scores[user, items], self.k)
        
        for metric in self.metrics:
            buff[f"{metric)}@{self.k}"] = buff[f"{metric)}@{self.k}"].mean()

        return buff

