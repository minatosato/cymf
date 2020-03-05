#
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++

import numpy as np

cimport cython
from cython.view cimport array
cimport numpy as np
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_lapack cimport dgesv as lapack_dgesv

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] zeros(int M, int N) nogil:
    cdef double[:,::1] ret
    with gil:
        ret = array(shape=(M, N), itemsize=sizeof(double), format="d")
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] zeros_like(double[:,::1] A) nogil:
    cdef double[:,::1] ret
    with gil:
        ret = array(shape=(A.shape[0], A.shape[1]), itemsize=sizeof(double), format="d")
    ret[:,:] = 0.0
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] eye(int N) nogil:
    cdef double[:,::1] ret
    cdef size_t i = 0
    with gil:
        ret = array(shape=(N, N), itemsize=sizeof(double), format="d")
    ret[:,:] = 0.0
    for i in range(N):
        ret[i,i] = 1.0
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void matmul(double alpha, double[:,:] A, double[:,:] B, double beta, double[:,:] C) nogil:
    """
    C = alpha * A   B   + beta * C
    """
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, C.shape[0], C.shape[1], B.shape[0],
                alpha, &A[0,0], A.shape[1], &B[0,0],
                B.shape[1], beta, &C[0,0], C.shape[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] broadcast_hadamard(double[:,::1] A, double[:,::1] b) nogil:
    """
    C = A * b
    """
    cdef double[:,::1] C = zeros_like(A)
    cdef int i, j
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i, j] = A[i, j] * b[i, 0]
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] atb_lambda(double alpha, double[:,::1] A, double[:,::1] B, double regularization) nogil:
    """
    C = alpha * AT B   + regularization * C
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[1] # Bの列数
    cdef int K = B.shape[0] #
    cdef double[:,::1] C = eye(M)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, M, N,
                K, alpha, &A[0,0], A.shape[1], &B[0,0],
                B.shape[1], regularization, &C[0,0], C.shape[1])
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] atbt(double[:,::1] A, double[:,::1] B) nogil:
    """
    C = AT BT
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[0] # BTの列数
    cdef int K = B.shape[1] #
    cdef double[:,::1] C = zeros(M, N)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans, M, N, K,
                1.0, &A[0,0], A.shape[1], &B[0,0], B.shape[1], 0.0, &C[0,0], C.shape[1])
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] atb(double[:,::1] A, double[:,::1] B) nogil:
    """
    C = AT B
    """
    cdef int M = A.shape[1] # ATの行数
    cdef int N = B.shape[1] # Bの列数
    cdef int K = B.shape[0] #
    cdef double[:,::1] C = zeros(M, N)
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, M, N, K,
                1.0, &A[0,0], A.shape[1], &B[0,0], B.shape[1], 0.0, &C[0,0], C.shape[1])
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dot(double[::1] x, double[::1] y) nogil:
    return cblas_ddot(x.shape[0], &x[0], 1, &y[0], 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int solve(double[::1,:] A, double[::1,:] b) nogil:
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
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int solvep(double* A, double* b, int K) nogil:
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
    cdef int n = K
    cdef int nrhs = 1
    cdef int* p_ipiv = <int*> malloc( n*sizeof(int) )
    cdef int info
    lapack_dgesv(&n, &nrhs, A, &n, p_ipiv, b, &n, &info)
    free( <void*>p_ipiv )
    return info