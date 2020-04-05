#
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_lapack cimport dgesv as lapack_dgesv

cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    double log(double x)
    double log2(double x)
    double pow(double x, double y)
    double fmin(double x, double y)
    const double M_PI

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T] nogil:
        uniform_int_distribution()
        uniform_int_distribution(T, T)
        T operator()(mt19937)

cdef inline double square(double x) nogil:
    return x * x

cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline double dmax(double x1, double x2) nogil:
    if x1 >= x2:
        return x1
    else:
        return x2

cdef class UniformGenerator(object):
    cdef mt19937 rng
    cdef uniform_int_distribution[long] uniform
    cdef inline long generate(self) nogil
