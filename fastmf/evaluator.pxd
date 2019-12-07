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

from .model cimport MfModel

cdef class Evaluator(object):
    cdef public MfModel model
    cpdef unordered_map[string, double] evaluate(self, np.ndarray[double, ndim=2] X, unordered_map[string, double] metrics, int num_negatives)
