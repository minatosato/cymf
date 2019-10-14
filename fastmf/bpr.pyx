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
import multiprocessing
import numpy as np
import pandas as pd
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

from .optimizer cimport Optimizer
from .optimizer cimport Adam
from .metrics import evaluate

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double log2(double x) nogil
    double sqrt(double x) nogil
    double pow(double x, double y) nogil

cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline floating square(floating x) nogil:
    return x * x

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

    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):

        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        num_threads = min(num_threads, multiprocessing.cpu_count())

        users, positives = utils.shuffle(*(X.nonzero()))
        dense = np.array(X.todense())

        users_valid = None
        positives_valid = None
        dense_valid = None
        if X_valid is not None:
            users_valid, positives_valid = X_valid.nonzero()
            dense_valid = np.array(X_valid.todense())

        return self._fit_bpr(users, 
                             positives,
                             dense,
                             users_valid,
                             positives_valid,
                             dense_valid,
                             X_test.toarray() if X_test is not None else None,
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
                 integral[:] users,
                 integral[:] positives,
                 np.ndarray[floating, ndim=2] X,
                 integral[:] users_valid,
                 integral[:] positives_valid,
                 np.ndarray[floating, ndim=2] X_valid,
                 np.ndarray[floating, ndim=2] X_test,
                 floating[:,:] W, 
                 floating[:,:] H, 
                 int num_iterations, 
                 floating learning_rate,
                 floating weight_decay,
                 int num_threads,
                 bool verbose):
        cdef int iterations = num_iterations
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, i, j, k, l, iteration
        cdef floating[:] loss = np.zeros(N)

        cdef unordered_map[string, double] metrics
        
        cdef list description_list

        cdef integral[:] negative_samples
        cdef integral[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)
        cdef integral[:,:] negatives_valid = None

        cdef vector[unordered_map[string, double]] history = []

        cdef BprModel bpr_model = BprModel(W, H, learning_rate, weight_decay, num_iterations)

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

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                metrics[b"loss"] = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = bpr_model.forward(users[l], positives[l], negatives[l, iteration])
                    bpr_model.backward(users[l], positives[l], negatives[l, iteration])

                for l in range(N):
                    metrics[b"loss"] += loss[l]
                metrics[b"loss"] /= N

                if X_valid is not None:
                    metrics[b"val_loss"] = 0.0
                    for l in prange(users_valid.shape[0], nogil=True, num_threads=num_threads):
                        loss[l] = bpr_model.forward(users_valid[l], positives_valid[l], negatives_valid[l, iteration])
                    for l in range(users_valid.shape[0]):
                        metrics[b"val_loss"] += loss[l]
                    metrics[b"val_loss"] /= users_valid.shape[0]

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(metrics[b'loss'], 4):.4f}")
                if X_valid is not None:
                    description_list.append(f"VAL_LOSS: {np.round(metrics[b'val_loss'], 4):.4f}")

                if X_test is not None:
                    metrics = evaluate(W, H, X_test, metrics)
                progress.set_description(', '.join(description_list))
                progress.update(1)

                history.push_back(metrics)
                
        df = pd.DataFrame(history)
        df.columns = list(map(lambda x: x.decode("utf-8"), df.columns))
        df.index += 1
        df.index.name = "epoch"
        return df

cdef class BprModel(object):
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef public int num_threads
    cdef public double[:] x_uij

    def __init__(self, double[:,:] W, double[:,:] H, double learning_rate, double weight_decay, int num_threads):
        self.W = W
        self.H = H        
        self.weight_decay = weight_decay
        self.num_threads = num_threads
        self.x_uij = np.zeros(self.num_threads)
        self.optimizer = Adam(learning_rate)
        self.optimizer.set_parameters(self.W, self.H)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double forward(self, int u, int i, int j) nogil:
        cdef int thread_id = threadid() # 自分のスレッドidを元に参照すべきdouble[:] x_uijを決定する

        cdef double tmp, loss, l2_norm
        cdef int K = self.W.shape[1]
        cdef int k

        self.x_uij[thread_id] = 0.0
        l2_norm = 0.0
        for k in range(K):
            self.x_uij[thread_id] += self.W[u, k] * (self.H[i, k] - self.H[j, k])
            l2_norm += square(self.W[u, k]) + square(self.H[i, k]) + square(self.H[j, k])

        loss = - log(sigmoid(self.x_uij[thread_id])) + self.weight_decay * l2_norm
        
        return loss

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void backward(self, int u, int i, int j) nogil:
        cdef int thread_id = threadid()

        cdef int N = self.W.shape[0]
        cdef int M = self.H.shape[0]
        cdef int K = self.W.shape[1]
        cdef int k

        cdef double grad_wuk
        cdef double grad_hik
        cdef double grad_hjk
        
        self.x_uij[thread_id] = (1.0 / (1.0 + exp(self.x_uij[thread_id])))

        for k in range(K):
            grad_wuk = - (self.x_uij[thread_id] * (self.H[i, k] - self.H[j, k]) - self.weight_decay * self.W[u, k])
            grad_hik = - (self.x_uij[thread_id] *  self.W[u, k] - self.weight_decay * self.H[i, k])
            grad_hjk = - (self.x_uij[thread_id] * (-self.W[u, k]) - self.weight_decay * self.H[j, k])

            self.optimizer.update_W(u, k, grad_wuk)
            self.optimizer.update_H(i, k, grad_hik)
            self.optimizer.update_H(j, k, grad_hjk)
                   
