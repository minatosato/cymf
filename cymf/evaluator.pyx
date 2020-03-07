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
from scipy import sparse
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set

from .metrics import dcg_at_k
from .metrics import recall_at_k
from .metrics import average_precision_at_k

from .metrics import dcg_at_k_with_ips
from .metrics import recall_at_k_with_ips
from .metrics import average_precision_at_k_with_ips

from .math cimport UniformGenerator

class Evaluator(object):
    def __init__(self, X,
                       X_train = None,
                       list metrics = ["DCG", "Recall", "MAP"],
                       k = 5,
                       int num_negatives = 100,
                       bool unbiased = False):
        self.X = sparse.csr_matrix(X)
        self.user_positives = self.X.copy()
        if X_train is not None:
            self.user_positives += sparse.csr_matrix(X_train)
        
        self.X = self.X.astype(np.float64)
        self.user_positives = self.user_positives.astype(np.float64)

        self.propensity_scores = np.maximum(X.mean(axis=0), 1e-3)
        self.metrics = metrics
        self.k = k
        self.num_negatives = num_negatives
        self.unbiased = unbiased

    def evaluate(self, np.ndarray W, np.ndarray H):
        cdef np.ndarray[double, ndim=2] _W = W.astype(np.float64)
        cdef np.ndarray[double, ndim=2] _H = H.astype(np.float64)
        cdef dict buff = {}
        cdef str metric
        cdef np.ndarray[int, ndim=1] positives
        cdef np.ndarray[int, ndim=1] negatives
        cdef vector[int] _negatives
        cdef int user, U
        cdef int i, I
        cdef int k
        cdef UniformGenerator gen
        cdef set[int] set_of_positives

        U = self.X.shape[0]
        I = self.X.shape[1]
        
        gen = UniformGenerator(0, I, seed=1234)

        if type(self.k) == int:
            self.k = [self.k]

        for k in self.k:
            for metric in self.metrics:
                buff[f"{metric)}@{k}"] = np.zeros(U)
        
        for user in range(U):
            positives = self.X[user].nonzero()[1]
            _negatives = []
            set_of_positives = {*self.user_positives[user].nonzero()[1]}
            for i in range(self.num_negatives):
                i = gen.generate()
                while set_of_positives.find(i) != set_of_positives.end():
                    i = gen.generate()
                _negatives.push_back(i)
            negatives = np.array(_negatives).astype(np.int32)

            items = np.r_[positives, negatives]
            ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)

            for k in self.k:
                for metric in self.metrics:
                    if self.unbiased:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k_with_ips(ratings, np.dot(_H[items], _W[user]), k, self.propensity_scores)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k_with_ips(ratings, np.dot(_H[items], _W[user]), k, self.propensity_scores)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k_with_ips(ratings, np.dot(_H[items], _W[user]), k, self.propensity_scores)
                    else:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k(ratings, np.dot(_H[items], _W[user]), k)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k(ratings, np.dot(_H[items], _W[user]), k)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k(ratings, np.dot(_H[items], _W[user]), k)
        
        for k in self.k:
            for metric in self.metrics:
                buff[f"{metric)}@{k}"] = buff[f"{metric)}@{k}"].mean()

        return buff

class AverageOverAllEvaluator(Evaluator):
    def __init__(self, X, X_train = None, list metrics = ["DCG", "Recall", "MAP"], k = 5, int num_negatives = 100):
        super(AverageOverAllEvaluator, self).__init__(X, X_train, metrics, k, num_negatives, unbiased=False)

AoaEvaluator = AverageOverAllEvaluator

class UnbiasedEvaluator(Evaluator):
    def __init__(self, X, X_train = None, list metrics = ["DCG", "Recall", "MAP"], k = 5, int num_negatives = 100):
        super(UnbiasedEvaluator, self).__init__(X, X_train, metrics, k, num_negatives, unbiased=True)
