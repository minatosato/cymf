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
import multiprocessing
import numpy as np
import pandas as pd
from collections import Counter
from cython.parallel import prange
from cython.parallel import threadid
from cython.operator cimport dereference
from cython.operator import postincrement
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from .linalg cimport solve
from .linalg cimport matmul
from .linalg cimport broadcast_hadamard
from .linalg cimport atb_lambda
from .linalg cimport atbt
from .linalg cimport atb
from .linalg cimport dot

cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    const double M_PI

cdef inline double square(double x) nogil:
    return x * x

cdef class ExpoMF(object):
    cdef public int num_components
    cdef public double weight_decay
    cdef public double[:,::1] W
    cdef public double[:,::1] H
    def __init__(self, int num_components,
                       double weight_decay = 0.01):
        self.num_components = num_components
        self.weight_decay = weight_decay

    def fit(self, X, X_valid = None, X_test = None,
                  int num_iterations = 10,
                  int num_threads = 8,
                  bool verbose = False):
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        self._fit(X, num_iterations, verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(self, np.ndarray[double, ndim=2] _X, int iterations, bool verbose):
        cdef double[:,::1] X = np.array(_X, dtype=np.float64, order="C")
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int K = self.num_components
        cdef int u, i, k, iteration
        cdef double[::1,:] A = np.zeros((K, K), order="F")
        cdef double[::1,:] b = np.zeros((K, 1), order="F")
        cdef double lam_y = self.weight_decay
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef double[:,::1] Exposure = np.zeros_like(_X)
        cdef double[:,::1] EY = np.zeros_like(_X)
        cdef double[:] mu = np.ones(I) * 0.01

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                for u in range(U):
                    for i in range(I):
                        if _X[u, i] == 1.0:
                            Exposure[u, i] = 1.0
                            EY[u, i] = 1.0 * lam_y
                        else:
                            n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- square(dot(self.W[u], self.H[i])) * square(lam_y))
                            Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / (mu[i]+1e-8))
                            EY[u, i] = Exposure[u, i] * X[u, i] * lam_y

                # Wの更新
                for u in range(U):
                    A = atb_lambda(lam_y, broadcast_hadamard(self.H, Exposure[u:u+1,:].T.copy()), self.H, self.weight_decay).copy_fortran()
                    b = atbt(self.H, EY[u:u+1, :]).copy_fortran()
                    solve(A, b)
                    for k in range(K):
                        self.W[u,k] = b[k,0]
                # Hの更新
                for i in range(I):
                    A = atb_lambda(lam_y, broadcast_hadamard(self.W, Exposure[:,i:i+1]), self.W, self.weight_decay).copy_fortran()
                    b = atb(self.W, EY[:,i:i+1].copy()).copy_fortran()
                    solve(A, b)
                    for k in range(K):
                        self.H[i, k] = b[k,0]
                # muの更新
                mu = (alpha_1 + np.array(Exposure).sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)

                progress.update(1)
