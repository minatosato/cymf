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
import multiprocessing
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from cython.parallel import prange
from cython.parallel import threadid
from cython.operator cimport dereference
from cython.operator import postincrement
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from .model cimport BprModel
from .optimizer cimport Optimizer
from .optimizer cimport Adam
from .evaluator cimport Evaluator
from .evaluator cimport UnbiasedEvaluator

cdef class BPR(object):
    cdef public int num_components
    cdef public double learning_rate
    cdef public double weight_decay
    cdef public double[:,:] W
    cdef public double[:,:] H
    def __init__(self, int num_components,
                       double learning_rate = 0.01,
                       double weight_decay = 0.01):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.W = None
        self.H = None

    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):

        if isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            X = X.toarray()
        X = X.astype(np.float64)
        
        if self.W is None:
            self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        if self.H is None:
            self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

        num_threads = min(num_threads, multiprocessing.cpu_count())

        users, positives = utils.shuffle(*(X.nonzero()))
        dense = np.array(X)

        users_valid = None
        positives_valid = None
        dense_valid = None
        if X_valid is not None:
            users_valid, positives_valid = X_valid.nonzero()
            dense_valid = np.array(X_valid.todense())

        users_test = None
        positives_test = None
        dense_test = None
        if X_test is not None:
            users_test, positives_test = X_test.nonzero()
            dense_test = np.array(X_test.todense())

        return self._fit_bpr(users.astype(np.int32), 
                             positives.astype(np.int32),
                             dense,
                             users_valid, # todo 修正
                             positives_valid, # todo 修正
                             dense_valid,
                             users_test, # todo 修正
                             positives_test, # todo 修正
                             dense_test,
                             self.W,
                             self.H,
                             num_iterations,
                             self.learning_rate,
                             self.weight_decay,
                             num_threads,
                             verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_bpr(self,
                 int[:] users,
                 int[:] positives,
                 np.ndarray[double, ndim=2] X,
                 int[:] users_valid,
                 int[:] positives_valid,
                 np.ndarray[double, ndim=2] X_valid,
                 int[:] users_test,
                 int[:] positives_test,
                 np.ndarray[double, ndim=2] X_test,
                 double[:,:] W, 
                 double[:,:] H, 
                 int num_iterations, 
                 double learning_rate,
                 double weight_decay,
                 int num_threads,
                 bool verbose):
        cdef int iterations = num_iterations
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, i, j, k, l, iteration
        cdef double[:] loss = np.zeros(N)

        cdef unordered_map[string, double] metrics
        cdef unordered_map[string, double] tmp
        
        cdef list description_list

        cdef int[:] negative_samples
        cdef int[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)
        cdef int[:,:] negatives_valid = None
        cdef int[:,:] negatives_test = None

        cdef vector[unordered_map[string, double]] history = []

        cdef Optimizer optimizer
        optimizer = Adam(learning_rate)
        optimizer.set_parameters(W, H)

        cdef BprModel bpr_model = BprModel(W, H, optimizer, weight_decay, num_threads)

        cdef np.ndarray[double, ndim=1] propensity_scores = X.sum(axis=0)
        propensity_scores[propensity_scores<1.0] = 1.0
        propensity_scores /= propensity_scores.max()
        propensity_scores = propensity_scores ** 0.5

        cdef UnbiasedEvaluator evaluator = UnbiasedEvaluator(bpr_model)
        #cdef Evaluator evaluator = Evaluator(bpr_model)

        for l in range(N):
            u = users[l]
            negative_samples = np.random.choice((X[u]-1).nonzero()[0], iterations).astype(np.int32)
            negatives[l][:] = negative_samples

        if X_valid is not None:
            negatives_valid = np.zeros((users_valid.shape[0], iterations)).astype(np.int32)
            for l in range(users_valid.shape[0]):
                u = users_valid[l]
                negative_samples = np.random.choice((X_valid[u]-1).nonzero()[0], iterations).astype(np.int32)
                negatives_valid[l][:] = negative_samples

        if X_test is not None:
            negatives_test = np.zeros((users_test.shape[0], iterations)).astype(np.int32)
            for l in range(users_test.shape[0]):
                u = users_test[l]
                negative_samples = np.random.choice((X_test[u]-1).nonzero()[0], iterations).astype(np.int32)
                negatives_test[l][:] = negative_samples

        with tqdm(total=iterations, leave=True, ncols=120, disable=not verbose) as progress:
            for iteration in range(iterations):
                metrics[b"train"] = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = bpr_model.forward(users[l], positives[l], negatives[l, iteration])
                    bpr_model.backward(users[l], positives[l], negatives[l, iteration])

                for l in range(N):
                    metrics[b"train"] += loss[l]
                metrics[b"train"] /= N

                if X_valid is not None:
                    metrics[b"valid"] = 0.0
                    for l in prange(users_valid.shape[0], nogil=True, num_threads=num_threads):
                        loss[l] = bpr_model.forward(users_valid[l], positives_valid[l], negatives_valid[l, iteration])
                    for l in range(users_valid.shape[0]):
                        metrics[b"valid"] += loss[l]
                    metrics[b"valid"] /= users_valid.shape[0]

                if X_test is not None:
                    metrics[b"test"] = 0.0
                    for l in prange(users_test.shape[0], nogil=True, num_threads=num_threads):
                        loss[l] = bpr_model.forward(users_test[l], positives_test[l], negatives_test[l, iteration])
                    for l in range(users_test.shape[0]):
                        metrics[b"test"] += loss[l]
                    metrics[b"test"] /= users_test.shape[0]

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(metrics[b'train'], 4):.4f}")
                if X_valid is not None:
                    description_list.append(f"VAL_LOSS: {np.round(metrics[b'valid'], 4):.4f}")

                if X_test is not None:
                    description_list.append(f"TEST_LOSS: {np.round(metrics[b'test'], 4):.4f}")

                if X_test is not None:
                    metrics = evaluator.evaluate(X_test, propensity_scores, metrics, 100)
                    #metrics = evaluator.evaluate(X_test, metrics, 100)
                progress.set_description(', '.join(description_list))
                progress.update(1)

                history.push_back(metrics)
                
        df = pd.DataFrame(history)
        df.columns = list(map(lambda x: x.decode("utf-8"), df.columns))
        df.index += 1
        df.index.name = "epoch"
        return df

