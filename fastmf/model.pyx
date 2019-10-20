
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

cdef class MfModel:
    def __init__(self, double[:,:] W, double[:,:] H, int num_threads):
        self.W = W
        self.H = H
        self.num_threads = num_threads
        self.tmp = np.zeros(self.num_threads)

cdef class BprModel(MfModel):
    def __init__(self, double[:,:] W, double[:,:] H, Optimizer optimizer, double weight_decay, int num_threads):
        super(BprModel, self).__init__(W, H, num_threads)
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double forward(self, int u, int i, int j) nogil:
        cdef int thread_id = threadid() # 自分のスレッドidを元に参照すべきdouble[:] x_uijを決定する

        cdef double loss, l2_norm
        cdef int K = self.W.shape[1]
        cdef int k

        self.tmp[thread_id] = 0.0
        l2_norm = 0.0
        for k in range(K):
            self.tmp[thread_id] += self.W[u, k] * (self.H[i, k] - self.H[j, k])
            l2_norm += square(self.W[u, k]) + square(self.H[i, k]) + square(self.H[j, k])

        loss = - log(sigmoid(self.tmp[thread_id])) + self.weight_decay * l2_norm
        
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
        
        self.tmp[thread_id] = (1.0 / (1.0 + exp(self.tmp[thread_id])))

        for k in range(K):
            grad_wuk = - (self.tmp[thread_id] * (self.H[i, k] - self.H[j, k]) - self.weight_decay * self.W[u, k])
            grad_hik = - (self.tmp[thread_id] *  self.W[u, k] - self.weight_decay * self.H[i, k])
            grad_hjk = - (self.tmp[thread_id] * (-self.W[u, k]) - self.weight_decay * self.H[j, k])

            self.optimizer.update_W(u, k, grad_wuk)
            self.optimizer.update_H(i, k, grad_hik)
            self.optimizer.update_H(j, k, grad_hjk)
                   