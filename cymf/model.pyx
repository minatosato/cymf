
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
from cython.parallel import prange
from sklearn import utils
from tqdm import tqdm

cimport numpy as np

from .optimizer cimport Optimizer

from .math cimport pow
from .math cimport fmin
from .math cimport square
from .math cimport sigmoid
from .math cimport log
from .math cimport exp

cdef extern from "util.h" namespace "cymf" nogil:
    cdef int threadid()

cdef inline double weight_func(double x, double x_max, double alpha) nogil:
    return fmin(pow(x / x_max, alpha), 1.0)

cdef class BprModel(object):
    def __init__(self, double[:,:] W, double[:,:] H, Optimizer optimizer, double weight_decay, int num_threads):
        self.W = W
        self.H = H
        self.tmp = np.zeros(num_threads)
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double forward(self, int u, int i, int j) nogil:
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
    cdef inline void backward(self, int u, int i, int j) nogil:
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


cdef class GloVeModel:
    def __init__(self,
                 double[:,:] W,
                 double[:,:] H,
                 double[:] central_bias,
                 double[:] context_bias,
                 double x_max,
                 double alpha,
                 GloVeAdaGrad optimizer,
                 int num_threads):
        self.W = W
        self.H = H
        self.central_bias = central_bias
        self.context_bias = context_bias
        self.x_max = x_max
        self.alpha = alpha
        self.optimizer = optimizer
        self.diff = np.zeros(num_threads)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double forward(self, int central, int context, double count) nogil:
        cdef int thread_id = threadid() # 自分のスレッドidを元に参照すべきdouble[:] diffを決定する

        cdef double tmp, loss
        cdef int K = self.W.shape[1]
        cdef int k

        self.diff[thread_id] = 0.0
        for k in range(K):
            self.diff[thread_id] += self.W[central, k] * self.H[context, k]
        self.diff[thread_id] += self.central_bias[central] + self.context_bias[context]
        self.diff[thread_id] -= log(count)
        tmp = self.diff[thread_id]
        self.diff[thread_id] *= weight_func(count, self.x_max, self.alpha)
        loss = 0.5 * self.diff[thread_id] * tmp
        return loss
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void backward(self, int central, int context) nogil:
        cdef int thread_id = threadid()

        cdef int K = self.W.shape[1]
        cdef int k
        cdef double grad_wck
        cdef double grad_hck
        cdef double grad_central_bias
        cdef double grad_context_bias

        for k in range(K):
            grad_wck = self.diff[thread_id] * self.H[context, k]
            grad_hck = self.diff[thread_id] * self.W[central, k]
            grad_central_bias = self.diff[thread_id]
            grad_context_bias = self.diff[thread_id]

            self.optimizer.update_W(central, k, grad_wck)
            self.optimizer.update_H(context, k, grad_hck)
            self.optimizer.update_bias_W(central, grad_central_bias)
            self.optimizer.update_bias_H(context, grad_context_bias)
