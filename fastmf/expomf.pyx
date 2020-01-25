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


cdef void matmul(double alpha, double[:,:] A, double[:,:] B, double beta, double[:,:] C):
    """
    C = alpha * A   B   + beta * C
    """
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, C.shape[0], C.shape[1],
               B.shape[0], alpha, &A[0,0], A.shape[1], &B[0,0],
               B.shape[1], beta, &C[0,0], C.shape[1])

cpdef double[:,::1] _broadcast_hadamard(double[:,::1] A, double[:,::1] b):
    """
    C = A * b
    """
    cdef double[:,::1] C = np.zeros_like(A).astype(np.float64)
    cdef int i, j
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i, j] = A[i, j] * b[i, 0]
    return C

cpdef double[:,::1] _atb_lambda(double alpha, double[:,::1] A, double[:,::1] B, double regularization):
    """
    C = alpha * AT B   + regularization * C
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[1] # Bの列数
    cdef int K = B.shape[0] #
    cdef double[:,::1] C = np.eye(M).astype(np.float64)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, M, N,
                K, alpha, &A[0,0], A.shape[1], &B[0,0],
                B.shape[1], regularization, &C[0,0], C.shape[1])
    return C


cpdef double[:,::1] _atbt(double[:,::1] A, double[:,::1] B):
    """
    C = AT BT
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[0] # BTの列数
    cdef int K = B.shape[1] #
    cdef double[:,::1] C = np.zeros((M, N))
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans, M, N,
               K, 1.0, &A[0,0], A.shape[1], &B[0,0],
               B.shape[1], 0.0, &C[0,0], C.shape[1])
    return C


cpdef double[:,::1] _atb(double[:,::1] A, double[:,::1] B):
    """
    C = AT B
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[1] # Bの列数
    cdef int K = B.shape[0] #
    cdef double[:,::1] C = np.zeros((M, N))
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, M, N,
               K, 1.0, &A[0,0], A.shape[1], &B[0,0],
               B.shape[1], 0.0, &C[0,0], C.shape[1])
    return C


cpdef double _dot(double[::1] x, double[::1] y):
    return cblas_ddot(x.shape[0], &x[0], 1, &y[0], 1)


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    const double M_PI
cdef inline floating square(floating x) nogil:
    return x * x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _solve(double[::1,:] A, double[::1,:] b) nogil:
    """
    倍精度実一般行列 A による連立一次方程式
    A x = b
    n: 連立一次方程式の式数 (= Aの行数)
    nrhs: bの列数
    A: 実行後は行列AをLU分解した結果が代入される．
    lda: n
    ipiv: 
    b: 実行後はxが入る．
    ldb: 通常はAの行数．
    info: 0なら正常終了．
    """
    cdef int n = A.shape[0]
    cdef int nrhs = b.shape[1]
    cdef int* p_ipiv = <int*> malloc( n*sizeof(int) )
    cdef int info
    lapack_dgesv(&n, &nrhs, &A[0,0], &n, p_ipiv, &b[0,0], &n, &info)
    free( <void*>p_ipiv )
    return info

cdef class ExpoMF(object):
    cdef public int num_components
    cdef public double weight_decay
    cdef public double[:,::1] W
    cdef public double[:,::1] H
    def __init__(self, int num_components,
                       double weight_decay = 0.01):
        self.num_components = num_components
        self.weight_decay = weight_decay
    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        self._fit(X, num_iterations)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, np.ndarray[double, ndim=2] _X, int num_iterations):
        cdef double[:,::1] X = np.array(_X, dtype=np.float64, order="C")
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int K = self.num_components
        cdef int u, i, k, iteration
        cdef double[::1,:] A = np.zeros((K, K), order="F")
        cdef double[::1,:] b = np.zeros((K, 1), order="F")
        cdef double lam_y = 0.01
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef double[:,::1] Exposure = np.zeros_like(_X)
        cdef double[:,::1] EY = np.zeros_like(_X)
        cdef double[:] mu = np.ones(I) * 0.01

        for iteration in range(num_iterations):
            for u in range(U):
                for i in range(I):
                    if _X[u, i] == 1.0:
                        Exposure[u, i] = 1.0
                        EY[u, i] = 1.0 * lam_y
                    else:
                        n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- square(_dot(self.W[u], self.H[i])) * square(lam_y))
                        Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / mu[i])
                        EY[u, i] = (n_ui / (n_ui + (1 - mu[i]) / mu[i])) * X[u, i] * lam_y

            # Wの更新
            for u in range(U):
                A = _atb_lambda(lam_y, _broadcast_hadamard(self.H, Exposure[u:u+1,:].T.copy()), self.H, self.weight_decay).copy_fortran()
                b = _atbt(self.H, EY[u:u+1, :]).copy_fortran()
                _solve(A, b)
                for k in range(K):
                    self.W[u,k] = b[k,0]
            # Hの更新
            for i in range(I):
                A = _atb_lambda(lam_y, _broadcast_hadamard(self.W, Exposure[:,i:i+1]), self.W, self.weight_decay).copy_fortran()
                b = _atb(self.W, EY[:,i:i+1].copy()).copy_fortran()
                _solve(A, b)
                for k in range(K):
                    self.H[i, k] = b[k,0]
            # muの更新
            mu = (alpha_1 + np.array(Exposure).sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)
