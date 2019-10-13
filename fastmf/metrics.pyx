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
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cimport cython


cdef extern from "math.h":
    double log2(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unordered_map[string, double] evaluate(double[:,:] W,
                                             double[:,:] H,
                                             np.ndarray[double, ndim=2] X,
                                             unordered_map[string, double] stores):
    cdef np.ndarray[double, ndim=2] scores = np.dot(np.array(W), np.array(H).T)
    cdef np.ndarray[int, ndim=2] argsorted_scores = scores.argsort(axis=1)[:,::-1][:,:10].astype(np.int32)
    stores[b"Recall@10"] = recall(argsorted_scores, X, k=10).mean()
    stores[b"nDCG@10"] = ndcg(argsorted_scores, X, k=10).mean()
    stores[b"MAP@10"] = ap(argsorted_scores, X, k=10).mean()
    return stores

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] recall(int[:,:] argsorted_scores, double[:,:] X, int k):
    cdef int N = X.shape[0]
    cdef double[:] _sum = np.array(X).sum(axis=1)
    cdef int i, j

    cdef double[:] ret = np.zeros(shape=(N,))

    for i in range(N):
        if _sum[i] == 0:
            ret[i] = 0.0
            continue
        for j in range(k):
            ret[i] += X[i, argsorted_scores[i, j]]
        ret[i] /= _sum[i]

    return np.array(ret)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] ndcg(int[:,:] argsorted_scores, double[:,:] X, int k):
    cdef int N = X.shape[0]
    cdef double[:] _sum = np.array(X).sum(axis=1)
    cdef int i, j
    cdef double[:] tmp = np.zeros(N)

    cdef double[:] ret = np.zeros(shape=(N,))

    for i in range(N):
        if _sum[i] == 0:
            ret[i] = 0.0
            continue

        tmp[i] = 1.0
        for j in range(1, <int>min(_sum[i], k)):
            tmp[i] += 1.0 / log2(j+1)

        ret[i] = X[i, argsorted_scores[i, 0]]
        for j in range(1, k):
            ret[i] += X[i, argsorted_scores[i, j]] / log2(j+1)

        ret[i] /= tmp[i]

    return np.array(ret)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] ap(int[:,:] argsorted_scores, double[:,:] X, int k):
    cdef int N = X.shape[0]
    cdef int i, j
    cdef double[:] counter = np.zeros(N)
    cdef double[:] rating = np.zeros(N)

    cdef double[:] ret = np.zeros(shape=(N,))

    for i in range(N):
        counter[i] = 0.0

        for j in range(k):
            rating[i] = X[i, argsorted_scores[i, j]]
            if rating[i] == 1.0:
                counter[i] += 1.0
                ret[i] += counter[i] / (j + 1)

        if counter[i] == 0.0:
            ret[i] = 0.0
            continue
        ret[i] /= counter[i]

    return np.array(ret)

    