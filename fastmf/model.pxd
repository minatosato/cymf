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

from .optimizer cimport Optimizer

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline floating square(floating x) nogil:
    return x * x

cdef class MfModel:
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double[:] tmp

cdef class BprModel(MfModel):
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef double forward(self, int u, int i, int j) nogil
    cdef void backward(self, int u, int i, int j) nogil

cdef class WmfModel(MfModel):
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef public double weight
    cdef double forward(self, int u, int i, double r) nogil
    cdef void backward(self, int u, int i) nogil

    