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
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cimport cython

from .math cimport log2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dcg_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]

    cdef int i
    cdef double dcg_tmp
    cdef double dcg_score = 0.0
    cdef double counter = 0.0

    dcg_tmp = <double> y_true_sorted_by_score[0]
    dcg_score += dcg_tmp

    for i in range(1, k):
        if 1 <= i < k:
            dcg_tmp = <double> y_true_sorted_by_score[i] / log2(<double>i+1.0)
            dcg_score += dcg_tmp
        
        counter += <double> y_true_sorted_by_score[i]

    if counter == 0.0:
        return 0.0
    
    return dcg_score / counter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dcg_at_k_with_ips(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k, np.ndarray[double, ndim=1] propensity_scores):
    cdef int[:] y_true_sorted_by_score = y_true[y_score.argsort()[::-1]]
    cdef double[:] p_scores_sorted_by_score = propensity_scores[y_score.argsort()[::-1]]

    cdef int i
    cdef double dcg_tmp
    cdef double dcg_score = 0.0
    cdef double sn = 0.0 # self normalizer

    dcg_tmp = <double> y_true_sorted_by_score[0] / p_scores_sorted_by_score[0]
    dcg_score += dcg_tmp

    for i in range(k):
        if 1 <= i < k:
            dcg_tmp = <double> y_true_sorted_by_score[i] / log2(<double>i+1.0) / p_scores_sorted_by_score[i]
            dcg_score += dcg_tmp
        
        sn += <double> y_true_sorted_by_score[i]  / p_scores_sorted_by_score[i]

    if sn == 0.0:
        return 0.0

    dcg_score /= sn
    return dcg_score

"""
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

    return average_precision_score / k
"""