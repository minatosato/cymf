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
from scipy.linalg.cython_lapack cimport dgesv, dgetrf, dgetrs
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport dger
from scipy.linalg.cython_blas cimport dgemm, dscal
cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    const double M_PI
cdef inline floating square(floating x) nogil:
    return x * x

cdef double[::1,:] _hadamard(double[::1,:] a, double[::1,:] b):
    cdef double[::1,:] c = np.zeros((a.shape[0], a.shape[1]))
    cdef int i
    for i in range(a.shape[0]):
        c[i, 0] = a[i, 0] * b[i, 0]
    return c
cdef double[::1,:] _xtx_lambda(double[::1,:] X, double regularization):
    """
    C := alpha * AB + beta * C
    """
    cdef int m = X.shape[1]
    cdef int n = X.shape[1]
    cdef int k = X.shape[0]
    cdef int ldA = X.shape[0]
    cdef int ldB = X.shape[0]
    cdef double[::1,:] C = np.eye(m, order="F")
    cdef int ldC = C.shape[0]
    cdef double alpha = 1.0
    cdef double beta = regularization
    dgemm("T", "N", &m, &n, &k, &alpha, &X[0,0], &ldA, &X[0,0], &ldB, &beta, &C[0,0], &ldC)
    return C
cdef double[::1,:] _matmul_atbt(double [::1,:] A, double [::1,:] B):
    cdef int m = A.shape[1]
    cdef int n = B.shape[0]
    cdef int k = A.shape[0]
    cdef int ldA = A.shape[0]
    cdef int ldB = B.shape[0]
    cdef double[::1,:] C = np.zeros((m, n), order="F")
    cdef int ldC = C.shape[0]
    cdef double alpha = 1.0
    cdef double beta = 0.0
    dgemm("T", "T", &m, &n, &k, &alpha, &A[0,0], &ldA, &B[0,0], &ldB, &beta, &C[0,0], &ldC)
    return C
cdef double[::1,:] _matmul_atb(double [::1,:] A, double [::1,:] B):
    cdef int m = A.shape[1]
    cdef int n = B.shape[1]
    cdef int k = A.shape[0]
    cdef int ldA = A.shape[0]
    cdef int ldB = B.shape[0]
    cdef double[::1,:] C = np.zeros((m, n), order="F")
    cdef int ldC = C.shape[0]
    cdef double alpha = 1.0
    cdef double beta = 0.0
    dgemm("T", "N", &m, &n, &k, &alpha, &A[0,0], &ldA, &B[0,0], &ldB, &beta, &C[0,0], &ldC)
    return C
cdef double[::1,:] _matmul_atb_lambda(double [::1,:] A, double [::1,:] B, double regularization):
    cdef int m = A.shape[1]
    cdef int n = B.shape[1]
    cdef int k = A.shape[0]
    cdef int ldA = A.shape[0]
    cdef int ldB = B.shape[0]
    cdef double[::1,:] C = np.zeros((m, n), order="F")
    cdef int ldC = C.shape[0]
    cdef double alpha = 1.0
    cdef double beta = regularization
    dgemm("T", "N", &m, &n, &k, &alpha, &A[0,0], &ldA, &B[0,0], &ldB, &beta, &C[0,0], &ldC)
    return C
def prod(A, x):
    return _product(np.array(A, order="F"), x)
cdef double[::1,:] _product(double[::1,:] A, double[:] x):
    cdef int i
    cdef int n = A.shape[1]
    cdef double alpha
    cdef double[:,:] _A = np.array(A).copy()
    for i in range(_A.shape[0]):
        alpha = x[i]
        _dscal(n, &_A[i][0], alpha)
    return _A.copy_fortran()
cdef void _dscal(int n, double *x, double alpha):
    cdef int inc_x = 1
    dscal(&n, &alpha, x, &inc_x)
cdef double[:,:] _matmul_ab(double [::1,:] A, double [::1,:] B):
    cdef int m = A.shape[0]
    cdef int n = B.shape[1]
    cdef int k = A.shape[1]
    cdef int ldA = A.shape[0]
    cdef int ldB = B.shape[0]
    cdef double[:,:] C = np.zeros((m, n), order="F")
    cdef int ldC = C.shape[0]
    cdef double alpha = 1.0
    cdef double beta = 0.0
    dgemm("N", "N", &m, &n, &k, &alpha, &A[0,0], &ldA, &B[0,0], &ldB, &beta, &C[0,0], &ldC)
    return C
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _dot(double[:] x, double[:] y) nogil:
    cdef int n = x.shape[0]
    cdef int incx = 1
    cdef int incy = 1
    return ddot(&n, &x[0], &incx, &y[0], &incy)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1,:] _abt(double[:] a, double[:] b):
    cdef int m = a.shape[0]
    cdef int n = b.shape[0]
    cdef double[::1,:] A = np.array(np.zeros(m, n), order="F")
    cdef double alpha = 1.0
    cdef int inc_a = 1
    cdef int inc_b = 1
    cdef int ldA = A.shape[0]
    dger(&m, &n, &alpha, &a[0], &inc_a, &b[0], &inc_b, &A[0, 0], &ldA)
    return A
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
    # solve (on exit, b is overwritten by the solution)
    #
    # http://www.netlib.org/lapack/lapack-3.1.1/html/dgesv.f.html
    # N, NRHS, A, LDA, IPIV, B, LDB, INFO
    #
    dgesv(&n, &nrhs, &A[0,0], &n, p_ipiv, &b[0,0], &n, &info)
    free( <void*>p_ipiv )
    return 0

cdef class ExpoMF(object):
    cdef public int num_components
    cdef public double weight_decay
    cdef public double[:,:] W
    cdef public double[:,:] H
    def __init__(self, int num_components,
                       double weight_decay = 0.01):
        self.num_components = num_components
        self.weight_decay = weight_decay
    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):
        self._fit(X, num_iterations)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, np.ndarray[double, ndim=2] X, int num_iterations):
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int K = self.num_components
        cdef int u, i, k, iteration
        cdef np.ndarray[double, ndim=2] A = np.zeros((K, K))
        cdef np.ndarray[double, ndim=2] b = np.zeros((K, 1))
        cdef double lam_y = 0.01
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef np.ndarray[double, ndim=2] Exposure = np.zeros_like(X)
        cdef double[:] mu = np.ones(I) * 0.01

        cdef np.ndarray[double, ndim=2] _W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        cdef np.ndarray[double, ndim=2] _H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

        for iteration in range(num_iterations):
            print("ITER:", iteration+1)
            for u in range(U):
                for i in range(I):
                    if X[u, i] == 1.0:
                        Exposure[u, i] = 1.0
                    else:
                        n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- (_W[u] @ _H[i])**2 * square(lam_y))
                        Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / mu[i])
            # Wの更新
            for u in range(U):
                A = lam_y * ((_H.T*Exposure[u:u+1,:]) @ _H) + self.weight_decay * np.eye(K)
                b = _H.T @ (X[u:u+1,:]*Exposure[u:u+1,:]).T * lam_y
                _W[u] = np.linalg.solve(A, b).flatten()

            # Hの更新
            for i in range(I):
                A = lam_y * ((_W.T*Exposure[:,i:i+1].T) @ _W) + self.weight_decay * np.eye(K)
                b = _W.T @ (X[:,i:i+1] * Exposure[:,i:i+1]) * lam_y
                _H[i] = np.linalg.solve(A, b).flatten()
            # muの更新
            mu = (alpha_1 + Exposure.sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)

            self.W = _W
            self.H = _H
"""
cdef class ExpoMF(object):
    cdef public int num_components
    cdef public double weight_decay
    cdef public double[::1,:] W
    cdef public double[::1,:] H
    def __init__(self, int num_components,
                       double weight_decay = 0.01):
        self.num_components = num_components
        self.weight_decay = weight_decay
    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):
        self.W = np.array(np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components, order="F", dtype=np.float64)
        self.H = np.array(np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components, order="F", dtype=np.float64)
        self._fit(X, num_iterations)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, np.ndarray[double, ndim=2] _X, int num_iterations):
        cdef double[::1,:] X = np.array(_X, dtype=np.float64, order="F")
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int K = self.num_components
        cdef int u, i, k, iteration
        cdef double[::1,:] A = np.zeros((K, K), order="F")
        cdef double[::1,:] b = np.zeros((K, 1), order="F")
        cdef double lam_y = 1.0
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef double[:,:] Exposure = np.zeros_like(_X)
        cdef double[:] mu = np.ones(I) * 0.01

        for iteration in range(num_iterations):
            print("ITER:", iteration+1)
            for u in range(U):
                for i in range(I):
                    if _X[u, i] == 1.0:
                        Exposure[u, i] = 1.0
                    else:
                        n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- square(_dot(self.W[u], self.H[i])) * square(lam_y))
                        Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / mu[i])
            # Wの更新
            for u in range(U):
                #A = _xtx_lambda(self.H, self.weight_decay)
                A = _matmul_atb_lambda(_product(self.H, Exposure[u]), self.H, self.weight_decay)
                b = _matmul_atbt(self.H, _hadamard(Exposure[u:u+1,:].copy_fortran(), X[u:u+1,:].copy_fortran()))
                _solve(A, b)
                self.W[u] = b[:, 1]
                for k in range(K):
                    self.W[u,k] = b[k,0]
            # Hの更新
            for i in range(I):
                #A = _xtx_lambda(self.W, self.weight_decay)
                A = _matmul_atb_lambda(_product(self.W, Exposure[:, i]), self.W, self.weight_decay)
                b = _matmul_atb(self.W, _hadamard(Exposure[:, i:i+1].copy_fortran(), X[:, i:i+1]))
                _solve(A, b)
                for k in range(K):
                    self.H[i,k] = b[k,0]
            # muの更新
            mu = (alpha_1 + np.array(Exposure).sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)
"""
"""
%%timeit
import fastmf
model = fastmf.expomf.ExpoMF(20, 0.01)
import numpy as np
from lightfm.datasets import fetch_movielens
dataset = fetch_movielens(min_rating=4.0)
Y = dataset["train"].toarray().astype(np.float64)
Y[Y > 0] = 1.0
model.fit(Y, num_iterations=3)
Y_test = dataset["test"].toarray()
Y_test[Y_test > 0] = 1.0
from sklearn import metrics
predicted = np.array(model.W) @ np.array(model.H).T
scores = np.zeros(Y_test.shape[0])
for u in range(Y_test.shape[0]):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test[u], predicted[u])
    scores[u] = metrics.auc(fpr, tpr) if len(set(Y_test[u])) != 1 else 0.0
print(f"test mean auc: {scores.mean()}")
"""
