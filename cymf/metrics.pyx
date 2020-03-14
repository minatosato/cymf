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
cpdef double dcg_at_k(int[:] y_true_sorted_by_score, int k) nogil:
    cdef int i
    cdef double dcg_tmp
    cdef double dcg_score = 0.0
    cdef double counter = 0.0

    dcg_tmp = <double> y_true_sorted_by_score[0]
    dcg_score += dcg_tmp

    for i in range(y_true_sorted_by_score.shape[0]):
        if 1 <= i < k:
            dcg_tmp = <double> y_true_sorted_by_score[i] / log2(<double>i+1.0)
            dcg_score += dcg_tmp
        
        counter += <double> y_true_sorted_by_score[i]

    if counter == 0.0:
        return 0.0
    
    return dcg_score / counter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dcg_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil:
    cdef int i
    cdef double dcg_tmp
    cdef double dcg_score = 0.0
    cdef double sn = 0.0 # self normalizer

    dcg_tmp = <double> y_true_sorted_by_score[0] / p_scores_sorted_by_score[0]
    dcg_score += dcg_tmp

    for i in range(y_true_sorted_by_score.shape[0]):
        if 1 <= i < k:
            dcg_tmp = <double> y_true_sorted_by_score[i] / log2(<double>i+1.0) / p_scores_sorted_by_score[i]
            dcg_score += dcg_tmp
        
        sn += <double> y_true_sorted_by_score[i]  / p_scores_sorted_by_score[i]

    if sn == 0.0:
        return 0.0

    dcg_score /= sn
    return dcg_score

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double recall_at_k(int[:] y_true_sorted_by_score, int k) nogil:
    cdef int i
    cdef double recall_score = 0.0
    cdef double counter = 0.0

    for i in range(y_true_sorted_by_score.shape[0]):
        if 0 <= i < k:
            recall_score += <double> y_true_sorted_by_score[i]
        
        counter += <double> y_true_sorted_by_score[i]

    if counter == 0.0:
        return 0.0

    return recall_score / counter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double recall_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil:
    cdef int i
    cdef double recall_score = 0.0
    cdef double sn = 0.0 # self normalizer

    for i in range(y_true_sorted_by_score.shape[0]):
        if 0 <= i < k:
            recall_score += <double> y_true_sorted_by_score[i] / p_scores_sorted_by_score[i]
        
        sn += <double> y_true_sorted_by_score[i] / p_scores_sorted_by_score[i]

    if sn == 0.0:
        return 0.0

    return recall_score / sn



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double average_precision_at_k(int[:] y_true_sorted_by_score, int k) nogil:
    cdef double average_precision_score = 0.0
    cdef double tmp
    cdef double counter = 0.0
    cdef int i

    for i in range(y_true_sorted_by_score.shape[0]):
        counter += <double> y_true_sorted_by_score[i]
        if 0 <= i < k:
            if y_true_sorted_by_score[i] == 1:
                average_precision_score +=  counter / (<double>i + 1.0)
    #average_precision_score /= k

    if counter == 0.0:
        return 0.0

    return average_precision_score / counter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double average_precision_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil:
    cdef double average_precision_score = 0.0
    cdef double tmp
    cdef double counter = 0.0
    cdef double sn = 0.0 # self normalizer
    cdef int i

    for i in range(y_true_sorted_by_score.shape[0]):
        counter += <double> y_true_sorted_by_score[i]
        sn += <double> y_true_sorted_by_score[i] / p_scores_sorted_by_score[i]
        if 0 <= i < k:
            if y_true_sorted_by_score[i] == 1:
                average_precision_score +=  sn / (<double>i + 1.0)
    #average_precision_score /= k

    if sn == 0.0:
        return 0.0

    return average_precision_score / sn
