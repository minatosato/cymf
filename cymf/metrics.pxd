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

cpdef double dcg_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k)
cpdef double dcg_at_k_with_ips(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k, np.ndarray[double, ndim=1] propensity_scores)

"""
cpdef double precision_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k)
cpdef double recall_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k)
cpdef double average_precision_at_k(np.ndarray[int, ndim=1] y_true, np.ndarray[double, ndim=1] y_score, int k)
"""
