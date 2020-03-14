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
from cython.parallel import prange
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

        self.propensity_scores = np.maximum(np.array(X.mean(axis=0)).flatten(), 1e-4)
        self.metrics = metrics
        self.k = k
        self.num_negatives = num_negatives
        self.unbiased = unbiased

    def evaluate(self, np.ndarray W, np.ndarray H, int seed = 1234):
        cdef np.ndarray[double, ndim=2] _W = W.astype(np.float64)
        cdef np.ndarray[double, ndim=2] _H = H.astype(np.float64)
        cdef dict buff = {}
        cdef str metric
        cdef int user, U
        cdef int item, I
        cdef int k
        cdef UniformGenerator gen
        cdef set[int] set_of_positives
        cdef int[:] sorted_predictions
        cdef np.ndarray[double, ndim=1] propensity_scores = self.propensity_scores

        cdef int[:] y_true_sorted_by_score
        cdef double[:] p_scores_sorted_by_score

        cdef int [:] indptr = self.X.indptr, indices = self.X.indices
        cdef int [:] all_indptr = self.user_positives.indptr, all_indices = self.user_positives.indices
        cdef int ptr
        cdef vector[int] items, feedbacks
        cdef int num_negatives = self.num_negatives

        U = self.X.shape[0]
        I = self.X.shape[1]
        
        gen = UniformGenerator(0, I, seed=seed)

        if type(self.k) == int:
            self.k = [self.k]

        for k in self.k:
            for metric in self.metrics:
                buff[f"{metric)}@{k}"] = np.zeros(U)

        for user in range(U):
            if indptr[user] == indptr[user+1]:
                continue

            items = vector[int]()
            feedbacks = vector[int]()

            for ptr in range(indptr[user], indptr[user+1]):
                items.push_back(indices[ptr])
                feedbacks.push_back(1)

            set_of_positives = set[int]()
            for ptr in range(all_indptr[user], all_indptr[user+1]):
                set_of_positives.insert(all_indices[ptr])

            for item in range(num_negatives):
                item = gen.generate()
                while set_of_positives.find(item) != set_of_positives.end():
                    item = gen.generate()
                items.push_back(item)
                feedbacks.push_back(0)

            sorted_predictions = np.dot(_H[np.array(items)], _W[user]).argsort()[::-1].astype(np.int32)
            y_true_sorted_by_score = np.array(feedbacks, dtype=np.int32)[sorted_predictions]
            if self.unbiased:
                p_scores_sorted_by_score = propensity_scores[sorted_predictions]

            for k in self.k:
                for metric in self.metrics:
                    if self.unbiased:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k_with_ips(y_true_sorted_by_score, p_scores_sorted_by_score, k)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k_with_ips(y_true_sorted_by_score, p_scores_sorted_by_score, k)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k_with_ips(y_true_sorted_by_score, p_scores_sorted_by_score, k)
                    else:
                        if metric == "DCG":
                            buff[f"{metric}@{k}"][user] = dcg_at_k(y_true_sorted_by_score, k)
                        elif metric == "Recall":
                            buff[f"{metric}@{k}"][user] = recall_at_k(y_true_sorted_by_score, k)
                        elif metric == "MAP":
                            buff[f"{metric}@{k}"][user] = average_precision_at_k(y_true_sorted_by_score, k)
        
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
