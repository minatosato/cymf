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
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map


cdef class Optimizer:
    def __init__(self, double[:,:] W, double [:,:] H):
        self.W = W
        self.H = H

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_W(self, int u, int k, double gradient) nogil:
        raise NotImplementedError()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_H(self, int i, int k, double gradient) nogil:
        raise NotImplementedError()

cdef class Sgd(Optimizer):
    def __init__(self, double[:,:] W, double [:,:] H, double learning_rate):
        super(Sgd, self).__init__(W, H)
        self.learning_rate = learning_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_W(self, int u, int k, double gradient) nogil:
        self.W[u, k] -= self.learning_rate * gradient

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_H(self, int i, int k, double gradient) nogil:
        self.H[i, k] -= self.learning_rate * gradient

cdef class AdaGrad(Sgd):
    def __init__(self, double[:,:] W, double [:,:] H, double learning_rate):
        super(AdaGrad, self).__init__(W, H, learning_rate)
        self.grad_accum_W = np.ones(shape=(W.shape[0], W.shape[1]))
        self.grad_accum_H = np.ones(shape=(H.shape[0], H.shape[1]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_W(self, int u, int k, double gradient) nogil:
        self.grad_accum_W[u, k] += square(gradient)
        self.W[u, k] -= self.learning_rate * gradient / sqrt(self.grad_accum_W[u, k])
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_H(self, int i, int k, double gradient) nogil:
        self.grad_accum_H[i, k] += square(gradient)
        self.H[i, k] -= self.learning_rate * gradient / sqrt(self.grad_accum_H[i, k])

cdef class Adam(Optimizer):
    def __init__(self,
                 double[:,:] W,
                 double[:,:] H,
                 double alpha = 0.001,
                 double beta1 = 0.9,
                 double beta2 = 0.999,
                 double epsilon = 1e-8
                 ):
        super(Adam, self).__init__(W, H)
        self.alpha = alpha
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.M_W = np.zeros(shape=(W.shape[0], W.shape[1]))
        self.V_W = np.zeros(shape=(W.shape[0], W.shape[1]))
        self.M_H = np.zeros(shape=(H.shape[0], H.shape[1]))
        self.V_H = np.zeros(shape=(H.shape[0], H.shape[1]))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_W(self, int u, int k, double gradient) nogil:
        self.M_W[u, k] = self.beta1 * self.M_W[u, k] + (1 - self.beta1) * gradient
        self.V_W[u, k] = self.beta2 * self.V_W[u, k] + (1 - self.beta2) * square(gradient)
        self.W[u, k] -= self.alpha * (self.M_W[u, k] / (1 - self.beta1)) / (sqrt(self.V_W[u, k] / (1 - self.beta2)) + self.epsilon)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_H(self, int i, int k, double gradient) nogil:
        self.M_H[i, k] = self.beta1 * self.M_H[i, k] + (1 - self.beta1) * gradient
        self.V_H[i, k] = self.beta2 * self.V_H[i, k] + (1 - self.beta2) * square(gradient)
        self.H[i, k] -= self.alpha * (self.M_H[i, k] / (1 - self.beta1)) / (sqrt(self.V_H[i, k] / (1 - self.beta2)) + self.epsilon)
        