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
from scipy import sparse
from cython.parallel import prange
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from libcpp cimport bool
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

from .linalg cimport solvep

cdef extern from "util.h" namespace "cymf" nogil:
    cdef int threadid()
    cdef int cpucount()

class WMF(object):
    """
    Weighted Matrix Factorization (WMF)
    http://yifanhu.net/PUB/cf.pdf
    
    Attributes:
        num_components (int): A dimensionality of latent vector
        weight_decay (double): A coefficient of weight decay
        weight (double): A weight for positive feedbacks.
        W (np.ndarray[double, ndim=2]): User latent vectors
        H (np.ndarray[double, ndim=2]): Item latent vectors
    """
    def __init__(self, int num_components = 20,
                       double weight_decay = 0.01,
                       double weight = 10.0):
        """
        Args:
            num_components (int): A dimensionality of latent vector
            weight_decay (double): A coefficient of weight decay
            weight (double): A weight for positive feedbacks.
        """
        self.num_components = num_components
        self.weight_decay = weight_decay
        self.weight = weight
        self.W = None
        self.H = None

    def fit(self, X, int num_epochs = 5, int num_threads = 1, bool verbose = True):
        """
        Training WMF model with ALS.

        Args:
            X: A user-item interaction matrix.
            num_epochs (int): A number of epochs.
            num_threads (int): A number of threads in HOGWILD! (http://i.stanford.edu/hazy/papers/hogwild-nips.pdf)
            verbose (bool): Whether to show the progress of training.
        """
        if X is None:
            raise ValueError()

        if sparse.isspmatrix(X):
            X = X.tocsr()
        elif isinstance(X, np.ndarray):
            X = sparse.csr_matrix(X)
        else:
            raise ValueError()
        X = X.astype(np.float64)
        
        if self.W is None:
            np.random.seed(4321)
            self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        if self.H is None:
            self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        self._fit_als(X, num_epochs, num_threads, verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_als(self,
                 X,
                 int num_epochs, 
                 int num_threads,
                 bool verbose):
        cdef int epoch
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]

        with tqdm(total=num_epochs, leave=True, ncols=100, disable=not verbose) as progress:
            for epoch in range(num_epochs):
                self._als(X.indptr, X.indices, self.W, self.H, num_threads)
                self._als(X.T.tocsr().indptr, X.T.tocsr().indices, self.H, self.W, num_threads)
                progress.set_description(f"EPOCH={epoch+1:{len(str(num_epochs))}}")
                progress.update(1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _als(self, int[:] indptr, int[:] indices, double[:,:] X, double[:,:] Y, int num_threads):
        cdef int K = X.shape[1]
        cdef int i, ptr
        cdef int index
        cdef int k, k2
        cdef double weight = self.weight
        cdef double[:,::1] YtY = np.dot(Y.T, Y)
        cdef double[::1,:] _A = YtY.copy_fortran() + (self.weight_decay * np.eye(K).astype(np.float64)).copy(order="F")
        cdef double[::1,:] _b = np.zeros((K, 1)).astype(np.float64)
        cdef double* A
        cdef double* b

        num_threads = num_threads if num_threads > 0 else cpucount()
        
        for i in prange(X.shape[0], nogil=True, num_threads=num_threads, schedule="guided"):
            A = <double *> malloc(sizeof(double) * K * K) # K行K列
            b = <double *> malloc(sizeof(double) * K * 1) # K行1列
            
            if indptr[i] == indptr[i+1]:
                memset(&X[i, 0], 0, sizeof(double) * K)
                continue
            
            memcpy(A, &_A[0, 0], sizeof(double) * K * K)
            memcpy(b, &_b[0, 0], sizeof(double) * K)
            
            for ptr in range(indptr[i], indptr[i+1]):
                index = indices[ptr]
                for k in range(K):
                    b[k] += Y[index, k] * weight
                    for k2 in range(K):
                        A[k*K+ k2] += Y[index, k] * Y[index, k2] * (weight-1.0)
            
            solvep(A, b, K)

            for k in range(K):
                X[i, k] = b[k]
            
            free(<void*> A)
            free(<void*> b)
