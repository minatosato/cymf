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
    