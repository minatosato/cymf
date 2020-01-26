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

cdef extern from 'cblas.h':
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans
    double cblas_ddot(int N, double* X, int incX, double* Y, int incY) nogil
    double cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                       int M, int N, int K,
                       double alpha, double *A, int lda, double *B, int ldb,
                       double beta, double *C, int ldc) nogil


cpdef void matmul(double alpha, double[:,:] A, double[:,:] B, double beta, double[:,:] C)
cpdef double[:,::1] broadcast_hadamard(double[:,::1] A, double[:,::1] b)
cpdef double[:,::1] atb_lambda(double alpha, double[:,::1] A, double[:,::1] B, double regularization)
cpdef double[:,::1] atbt(double[:,::1] A, double[:,::1] B)
cpdef double[:,::1] atb(double[:,::1] A, double[:,::1] B)
cpdef double dot(double[::1] x, double[::1] y)
cpdef int solve(double[::1,:] A, double[::1,:] b) nogil
