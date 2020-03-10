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

from .optimizer cimport Optimizer
from .optimizer cimport GloVeAdaGrad

cdef class BprModel(object):
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double[:] tmp
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef inline double forward(self, int u, int i, int j) nogil
    cdef inline void backward(self, int u, int i, int j) nogil

cdef class RelMfModel(object):
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double[:] tmp
    cdef public double weight_decay
    cdef public Optimizer optimizer
    cdef inline double forward(self, int u, int i, double r, double p, double M) nogil
    cdef inline void backward(self, int u, int i, double r, double p, double M) nogil

cdef class GloVeModel(object):
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public GloVeAdaGrad optimizer
    cdef public double[:] diff

    cdef public double[:] central_bias
    cdef public double[:] context_bias
    cdef public double x_max
    cdef public double alpha

    cdef double forward(self, int central, int context, double count) nogil
    cdef void backward(self, int central, int context) nogil
