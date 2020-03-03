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
from tqdm import tqdm

cimport numpy as np
from cython.parallel cimport prange
from libcpp cimport bool

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

from .linalg cimport solvep
from .linalg cimport solve
from .linalg cimport broadcast_hadamard
from .linalg cimport atb_lambda
from .linalg cimport atbt
from .linalg cimport atb
from .linalg cimport dot

from .math cimport sqrt
from .math cimport exp
from .math cimport square
from .math cimport M_PI

class ExpoMF(object):
    """
    Exposure Matrix Factorization (ExpoMF)
    https://arxiv.org/pdf/1510.07025.pdf
    
    Attributes:
        num_components (int): A dimensionality of latent vector
        lam_y (double): See the paper
        weight_decay (double): A coefficient of weight decay
        W (np.ndarray[double, ndim=2]): User latent vectors
        H (np.ndarray[double, ndim=2]): Item latent vectors
    """
    def __init__(self, int num_components = 20, double lam_y = 1.0, double weight_decay = 0.01):
        """
        Args:
            num_components (int): A dimensionality of latent vector
            weight_decay (double): A coefficient of weight decay
        """
        self.num_components = num_components
        self.lam_y = lam_y
        self.weight_decay = weight_decay
        self.W = None
        self.H = None

    def fit(self, X, int num_epochs = 10, bool verbose = False):
        """
        Training ExpoMF model with EM Algorithm

        Args:
            X: A user-item interaction matrix.
            num_epochs (int): A number of epochs.
            verbose (bool): Whether to show the progress of training.
        """
        if X is None:
            raise ValueError()
        if isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            X = X.toarray()
        X = X.astype(np.float64)
        
        if self.W is None:
            np.random.seed(4321)
            self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        if self.H is None:
            self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        self.__fit(X, num_epochs, verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __fit(self, np.ndarray[double, ndim=2] _X, int iterations, bool verbose):
        cdef double[:,::1] X = np.array(_X, dtype=np.float64, order="C")
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int K = self.num_components
        cdef int u, i, k, iteration
        cdef double[::1,:] A = np.zeros((K, K), order="F")
        cdef double[::1,:] b = np.zeros((K, 1), order="F")
        cdef double lam_y = self.lam_y
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef double[:,::1] Exposure = np.zeros_like(_X)
        cdef double[:,::1] EY = np.zeros_like(_X)
        cdef double[:] mu = np.ones(I) * 0.01

        cdef double[:,::1] W = self.W
        cdef double[:,::1] H = self.H

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                for u in range(U):
                    for i in range(I):
                        if _X[u, i] == 1.0:
                            Exposure[u, i] = 1.0
                            EY[u, i] = 1.0 * lam_y
                        else:
                            n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- square(dot(W[u], H[i])) * square(lam_y))
                            Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / (mu[i]+1e-8))
                            EY[u, i] = Exposure[u, i] * X[u, i] * lam_y

                # Wの更新
                for u in range(U):
                    A = atb_lambda(lam_y, broadcast_hadamard(H, Exposure[u:u+1,:].T.copy()), H, self.weight_decay).copy_fortran()
                    b = atbt(H, EY[u:u+1, :]).copy_fortran()
                    solve(A, b)
                    for k in range(K):
                        W[u,k] = b[k,0]
                # Hの更新
                for i in range(I):
                    A = atb_lambda(lam_y, broadcast_hadamard(W, Exposure[:,i:i+1]), W, self.weight_decay).copy_fortran()
                    b = atb(W, EY[:,i:i+1].copy()).copy_fortran()
                    solve(A, b)
                    for k in range(K):
                        H[i, k] = b[k,0]
                # muの更新
                mu = (alpha_1 + np.array(Exposure).sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)

                progress.update(1)

        self.W = np.array(W)
        self.H = np.array(H)

    def fit_als(self, X, int num_epochs, int num_threads, bool verbose = True):
        if X is None:
            raise ValueError()

        if not isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            X = sparse.csr_matrix(X)
        else:
            X = X.tocsr()
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
        cdef list description_list

        cdef double[:,:] Exposure = np.zeros(shape=X.shape, dtype=np.float64)
        cdef double[:,:] EY = np.zeros(shape=X.shape, dtype=np.float64)
        cdef double alpha_1 = 1.0
        cdef double alpha_2 = 1.0
        cdef double[:] mu = np.ones(I) * 0.01
        cdef double lam_y = self.lam_y
        cdef int u, i
        cdef double n_ui
        cdef double[:,::1] W = self.W
        cdef double[:,::1] H = self.H

        with tqdm(total=num_epochs, leave=True, ncols=100, disable=not verbose) as progress:
            for epoch in range(num_epochs):

                for u in range(U):
                    for i in range(I):
                        if X[u, i] == 1.0:
                            Exposure[u, i] = 1.0
                            EY[u, i] = 1.0 * lam_y
                        else:
                            n_ui = (lam_y / sqrt(2.0*M_PI)) * exp(- square(dot(W[u], H[i])) * square(lam_y))
                            Exposure[u, i] = n_ui / (n_ui + (1 - mu[i]) / (mu[i]+1e-8))
                            EY[u, i] = Exposure[u, i] * X[u, i] * lam_y

                self._als(X.indptr, X.indices, Exposure, EY, self.W, self.H, num_threads)
                self._als(X.T.tocsr().indptr, X.T.tocsr().indices, Exposure, EY.T, self.H, self.W, num_threads)

                mu = (alpha_1 + np.array(Exposure).sum(axis=0) - 1.) / (alpha_1 + alpha_2 + U - 2.)

                description_list = []
                description_list.append(f"EPOCH={epoch+1:{len(str(num_epochs))}}")
                progress.set_description(", ".join(description_list))
                progress.update(1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _als(self, int[:] indptr, int[:] indices, double[:,:] Exposure, double[:,:] EY, double[:,:] X, double[:,:] Y, int num_threads):
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
        
        for i in prange(X.shape[0], nogil=True, num_threads=num_threads):
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
                    b[k] += Y[index, k] * EY[index, k]
                    for k2 in range(K):
                        A[k*K+ k2] += Y[index, k] * Y[index, k2] * (weight-1.0)
            
            solvep(A, b, K)

            for k in range(K):
                X[i, k] = b[k]
            
            free(<void*> A)
            free(<void*> b)
