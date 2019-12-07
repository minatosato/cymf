# 
# Copyright (c) 2019 Minato Sato
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

from .model cimport MfModel
from .metrics import dcg_at_k
from .metrics import recall_at_k
from .metrics import average_precision_at_k

from .metrics import dcg_at_k_with_ips
#from .metrics import recall_at_k_with_ips
#from .metrics import average_precision_at_k_with_ips

cdef class Evaluator(object):
    def __init__(self, MfModel model):
        self.model = model


    cpdef unordered_map[string, double] evaluate(self, np.ndarray[double, ndim=2] X, unordered_map[string, double] metrics, int num_negatives):
        cdef np.ndarray[double, ndim=2] scores = np.dot(np.array(self.model.W), np.array(self.model.H).T)
        cdef dict buff = {}


        mapper = {
            "MAP": average_precision_at_k,
            "Recall": recall_at_k,
            "DCG": dcg_at_k
        }

        cdef list k = [10]

        cdef str key
        cdef int _k

        for key in mapper.keys():
            for _k in k:
                buff[f"{key}@{_k}"] = np.zeros(self.model.W.shape[0])
        
        cdef np.ndarray[np.int_t, ndim=1] positives
        cdef np.ndarray[np.int_t, ndim=1] negatives
        cdef int user
        for user in range(X.shape[0]):
            positives = X[user].nonzero()[0]
            negatives = np.random.permutation((X[user]==0.).nonzero()[0])[:num_negatives]

            items = np.r_[positives, negatives]
            ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)
            
            for key in mapper.keys():
                for _k in k:
                    buff[f"{key}@{_k}"][user] = mapper[key](ratings, scores[user, items], _k)
        
        for key in mapper.keys():
            for _k in k:
                metrics[(f"{key}@{_k}").encode("utf-8")] = buff[f"{key}@{_k}"].mean()

        return metrics



cdef class UnbiasedEvaluator(object):
    def __init__(self, MfModel model):
        self.model = model

    cpdef unordered_map[string, double] evaluate(self, np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] propensity_scores, unordered_map[string, double] metrics, int num_negatives):
        cdef np.ndarray[double, ndim=2] scores = np.dot(np.array(self.model.W), np.array(self.model.H).T)
        cdef dict buff = {}


        mapper = {
            #"Recall": recall_at_k_with_ips,
            #"MAP": average_precision_at_k_with_ips,
            "DCG": dcg_at_k_with_ips
        }

        cdef list k = [10]

        cdef str key
        cdef int _k

        for key in mapper.keys():
            for _k in k:
                buff[f"{key}@{_k}"] = np.zeros(self.model.W.shape[0])
        
        cdef np.ndarray[np.int_t, ndim=1] positives
        cdef np.ndarray[np.int_t, ndim=1] negatives
        cdef int user
        for user in range(X.shape[0]):
            positives = X[user].nonzero()[0]
            negatives = np.random.permutation((X[user]==0.).nonzero()[0])[:num_negatives]

            items = np.r_[positives, negatives]
            ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)
            
            for key in mapper.keys():
                for _k in k:
                    buff[f"{key}@{_k}"][user] = mapper[key](ratings, scores[user, items], propensity_scores[items], _k)
        
        for key in mapper.keys():
            for _k in k:
                metrics[(f"{key}@{_k}").encode("utf-8")] = buff[f"{key}@{_k}"].mean()

        return metrics

