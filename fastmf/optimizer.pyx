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

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map

from .math cimport sqrt
from .math cimport square

cdef class Optimizer:
    def __init__(self):
        raise NotImplementedError()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_parameters(self, double[:,:] W, double[:,:] H):
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
    def __init__(self, double learning_rate):
        self.learning_rate = learning_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_parameters(self, double[:,:] W, double[:,:] H):
        self.W = W
        self.H = H

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_W(self, int u, int k, double gradient) nogil:
        self.W[u, k] -= self.learning_rate * gradient

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_H(self, int i, int k, double gradient) nogil:
        self.H[i, k] -= self.learning_rate * gradient

cdef class AdaGrad(Sgd):
    def __init__(self, double learning_rate):
        super(AdaGrad, self).__init__(learning_rate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_parameters(self, double[:,:] W, double[:,:] H):
        self.W = W
        self.H = H
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


cdef class GloVeAdaGrad:
    def __init__(self, double learning_rate):
        self.learning_rate = learning_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_parameters(self, double[:,:] W, double[:,:] H, double[:] bias_W, double[:] bias_H):
        self.W = W
        self.H = H
        self.bias_W = bias_W
        self.bias_H = bias_H
        self.grad_accum_W = np.ones(shape=(W.shape[0], W.shape[1]))
        self.grad_accum_H = np.ones(shape=(H.shape[0], H.shape[1]))
        self.grad_accum_bias_W = np.ones(shape=(bias_W.shape[0]))
        self.grad_accum_bias_H = np.ones(shape=(bias_H.shape[0]))

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_bias_W(self, int u, double gradient) nogil:
        self.grad_accum_bias_W[u] += square(gradient)
        self.bias_W[u] -= self.learning_rate * gradient / sqrt(self.grad_accum_bias_W[u])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_bias_H(self, int i, double gradient) nogil:
        self.grad_accum_bias_H[i] += square(gradient)
        self.bias_H[i] -= self.learning_rate * gradient / sqrt(self.grad_accum_bias_H[i])


cdef class Adam(Optimizer):
    def __init__(self,
                 double alpha = 0.001,
                 double beta1 = 0.9,
                 double beta2 = 0.999,
                 double epsilon = 1e-8
                 ):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_parameters(self, double[:,:] W, double[:,:] H):
        self.W = W
        self.H = H
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
        