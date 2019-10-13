# 
# Copyright (c) 2019 Minato Sato
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


cpdef unordered_map[string, double] evaluate(double[:,:] W, double[:,:] H, np.ndarray[double, ndim=2] X, unordered_map[string, double] stores)
cpdef np.ndarray[np.double_t, ndim=1] recall(int[:,:] argsorted_scores, double[:,:] X, int k)
cpdef np.ndarray[np.double_t, ndim=1] ndcg(int[:,:] argsorted_scores, double[:,:] X, int k)
cpdef np.ndarray[np.double_t, ndim=1] ap(int[:,:] argsorted_scores, double[:,:] X, int k)
