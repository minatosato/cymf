# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++

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

cdef class GloVeAdaGrad:
    cdef public double learning_rate
    cdef public double[:,:] W
    cdef public double[:,:] H
    cdef public double[:] bias_W
    cdef public double[:] bias_H
    cdef public double[:,:] grad_accum_W
    cdef public double[:,:] grad_accum_H
    cdef public double[:] grad_accum_bias_W
    cdef public double[:] grad_accum_bias_H
    cdef void set_parameters(self, double[:,:] W, double[:,:] H, double[:] bias_W, double[:] bias_H)
    cdef void update_W(self, int u, int k, double gradient) nogil
    cdef void update_H(self, int i, int k, double gradient) nogil
    cdef void update_bias_W(self, int u, double gradient) nogil
    cdef void update_bias_H(self, int i, double gradient) nogil

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
