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

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double pow(double x, double y) nogil

cdef inline floating square(floating x) nogil:
    return x * x

cdef class WMF(object):
    cdef public int num_components
    cdef public double learning_rate
    cdef public double weight_decay
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double weight
    def __init__(self, int num_components,
                       double learning_rate = 0.01,
                       double weight_decay = 0.01,
                       double weight = 5.0):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight = weight

    def fit(self, X, 
                  int num_iterations,
                  int num_threads,
                  bool verbose = False):
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

        tmp = pd.DataFrame(X.todense()).stack().reset_index()
        tmp.columns = ("user", "item", "rating")
        users = tmp["user"].values
        items = tmp["item"].values
        ratings = tmp["rating"].values

        users, items, ratings = utils.shuffle(users, items, ratings)

        users = users.astype(np.int32)
        items = items.astype(np.int32)
        num_threads = min(num_threads, multiprocessing.cpu_count())
        self._fit_wmf(users, 
                     items,
                     ratings,
                     self.W,
                     self.H,
                     num_iterations,
                     self.learning_rate,
                     self.weight_decay,
                     self.weight,
                     num_threads,
                     verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_wmf(self,
                 integral[:] users,
                 integral[:] items,
                 floating[:] ratings,
                 floating[:,:] W, 
                 floating[:,:] H, 
                 int num_iterations, 
                 floating learning_rate,
                 floating weight_decay,
                 floating weight,
                 int num_threads,
                 bool verbose):
        cdef int iterations = num_iterations
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, i, j, k, l, iteration
        cdef floating[:] prediction = np.zeros(N)
        cdef floating[:] w_uk = np.zeros(N)
        cdef floating[:] l2_norm = np.zeros(N)
        cdef floating[:] diff = np.zeros(N)
        cdef floating[:] loss = np.zeros(N)
        cdef floating acc_loss
        
        cdef list description_list

        cdef WmfModel wmf_model = WmfModel(W, H, learning_rate, weight_decay, num_threads, weight)

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                acc_loss = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = wmf_model.forward(users[l], items[l], ratings[l])
                    wmf_model.backward(users[l], items[l])

                for l in range(N):                
                    acc_loss += loss[l]

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(acc_loss/N, 4):.4f}")
                progress.set_description(", ".join(description_list))
                progress.update(1)

cdef class WmfModel(object):
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef public int num_threads
    cdef public double[:] diff
    cdef public double weight

    def __init__(self, double[:,:] W, double[:,:] H, double learning_rate, double weight_decay, int num_threads, double weight):
        self.W = W
        self.H = H        
        self.weight_decay = weight_decay
        self.num_threads = num_threads
        self.diff = np.zeros(self.num_threads)
        self.weight = weight
        self.optimizer = Adam(self.W, self.H, learning_rate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double forward(self, int u, int i, double r) nogil:
        cdef int thread_id = threadid()
        cdef int K = self.W.shape[1]
        cdef int k
        cdef double loss, l2_norm

        self.diff[thread_id] = 0.0
        l2_norm = 0.0
        for k in range(K):
            self.diff[thread_id] += self.W[u, k] * self.H[i, k]
            l2_norm += square(self.W[u, k]) + square(self.H[i, k])
        self.diff[thread_id] = r - self.diff[thread_id]
        if r != 0.0:
            self.diff[thread_id] *= self.weight

        loss = square(self.diff[thread_id]) + self.weight_decay * l2_norm
        return loss

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void backward(self, int u, int i) nogil:
        cdef int thread_id = threadid()
        cdef int K = self.W.shape[1]
        cdef int k
        cdef double grad_wuk
        cdef double grad_hik

        for k in range(K):
            grad_wuk = - self.diff[thread_id] * self.H[i, k]
            grad_hik = - self.diff[thread_id] * self.W[u, k]

            self.optimizer.update_W(u, k, grad_wuk)
            self.optimizer.update_H(i, k, grad_hik)
