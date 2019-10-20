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

from .model cimport WmfModel
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
                 double[:] ratings,
                 double[:,:] W, 
                 double[:,:] H, 
                 int num_iterations, 
                 double learning_rate,
                 double weight_decay,
                 double weight,
                 int num_threads,
                 bool verbose):
        cdef int iterations = num_iterations
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, i, j, k, l, iteration
        cdef double[:] prediction = np.zeros(N)
        cdef double[:] w_uk = np.zeros(N)
        cdef double[:] l2_norm = np.zeros(N)
        cdef double[:] diff = np.zeros(N)
        cdef double[:] loss = np.zeros(N)
        cdef double acc_loss
        
        cdef list description_list

        cdef Optimizer optimizer
        optimizer = Adam(learning_rate)
        optimizer.set_parameters(W, H)
        cdef WmfModel wmf_model = WmfModel(W, H, optimizer, weight_decay, num_threads, weight)

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

