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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dcg_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    cdef double dcg_score = y_true_sorted_by_score[0]
    cdef int i
    cdef double total = 0.0

    for i in range(y_true.shape[0]):
        if 1 <= i < k:
            dcg_score += <double> y_true_sorted_by_score[i] / log2(<double>i+1.0)
        total += <double> y_true_sorted_by_score[i]

    if total == 0.0:
        return 0.0

    return dcg_score / total

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double precision_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]][:k]
    cdef double precision_score = 0.0
    cdef int i

    for i in range(y_true.shape[0]):
        precision_score += y_true_sorted_by_score[i]

    precision_score /= k
    return precision_score


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double recall_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]
    cdef int i

    cdef int num_positive_items_in_k = 0
    cdef int num_positive_items = 0
    for i in range(y_true.shape[0]):
        num_positive_items += y_true_sorted_by_score[i]
        if i < k:
            num_positive_items_in_k += y_true_sorted_by_score[i]
    
    if num_positive_items == 0:
        return 0.0
    
    return num_positive_items_in_k / num_positive_items

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double average_precision_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]
    cdef double average_precision_score = 0.0

    cdef double total = 0.
    cdef int i
    cdef double count = 0
    cdef int num_items = y_true.shape[0]
    for i in range(num_items):
        total += <double>y_true_sorted_by_score[i]

    if total == 0:
        return 0.0

    for i in range(k):
        if y_true_sorted_by_score[i] == 1:
            count += 1.0
            average_precision_score += count / (<double>(i + 1))
    
    #if count == 0.0:
    #    return 0.0

    return average_precision_score / total
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unordered_map[string, double] eval_test(double[:,:] W, double[:,:] H, np.ndarray[double, ndim=2] X, unordered_map[string, double] metrics, int num_negatives):
    cdef np.ndarray[double, ndim=2] scores = np.dot(np.array(W), np.array(H).T)

    cdef dict buff = {}

    cdef dict mapper = {
        "MAP": average_precision_at_k,
        "Recall": recall_at_k,
        "DCG": dcg_at_k
    }

    cdef list k = [10]

    cdef str key
    cdef int _k

    for key in mapper.keys():
        for _k in k:
            buff[f"{key}@{_k}"] = np.zeros(W.shape[0])
    
    cdef np.ndarray[np.int_t, ndim=1] positives
    cdef np.ndarray[np.int_t, ndim=1] negatives
    cdef int user
    for user in range(X.shape[0]):
        positives = X[user].nonzero()[0]
        negatives = np.random.permutation((X[user]==0.).nonzero()[0])[:num_negatives]

        items = np.r_[positives, negatives]
        ratings = np.r_[np.ones_like(positives), np.zeros_like(negatives)].astype(np.int32)
        
        for key in mapper.keys():
            for _k in k:
                buff[f"{key}@{_k}"][user] = mapper[key](ratings, scores[user, items], _k)
    
    for key in mapper.keys():
        for _k in k:
            metrics[(f"{key}@{_k}").encode("utf-8")] = buff[f"{key}@{_k}"].mean()

    return metrics    

