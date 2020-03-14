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
cimport numpy as np
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cpdef double dcg_at_k(int[:] y_true_sorted_by_score, int k) nogil
cpdef double dcg_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil
cpdef double recall_at_k(int[:] y_true_sorted_by_score, int k) nogil
cpdef double recall_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil
cpdef double average_precision_at_k(int[:] y_true_sorted_by_score, int k) nogil
cpdef double average_precision_at_k_with_ips(int[:] y_true_sorted_by_score, double[:] p_scores_sorted_by_score, int k) nogil
