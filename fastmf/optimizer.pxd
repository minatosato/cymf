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

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map


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

cdef class Optimizer:
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef void set_parameters(self, double[:,:] W, double[:,:] H)
    cdef void update_W(self, int u, int k, double gradient) nogil
    cdef void update_H(self, int i, int k, double gradient) nogil

cdef class Sgd(Optimizer):
    cdef public double learning_rate
    cdef void set_parameters(self, double[:,:] W, double[:,:] H)
    cdef void update_W(self, int u, int k, double gradient) nogil
    cdef void update_H(self, int i, int k, double gradient) nogil

cdef class AdaGrad(Sgd):
    cdef public double[:,:] grad_accum_W
    cdef public double[:,:] grad_accum_H
    cdef void set_parameters(self, double[:,:] W, double[:,:] H)
    cdef void update_W(self, int u, int k, double gradient) nogil
    cdef void update_H(self, int i, int k, double gradient) nogil

cdef class Adam(Optimizer):
    cdef public double alpha
    cdef public double beta1
    cdef public double beta2
    cdef public double epsilon
    cdef public double[:,:] M_W
    cdef public double[:,:] V_W
    cdef public double[:,:] M_H
    cdef public double[:,:] V_H
    cdef void set_parameters(self, double[:,:] W, double[:,:] H)
    cdef void update_W(self, int u, int k, double gradient) nogil
    cdef void update_H(self, int i, int k, double gradient) nogil
