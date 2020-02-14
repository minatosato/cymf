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
from libcpp cimport bool

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

    def fit(self, X, int num_iterations = 10, bool verbose = False):
        """
        Training ExpoMF model with EM Algorithm

        Args:
            X: A user-item interaction matrix.
            num_iterations (int): A number of epochs.
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
        self.__fit(X, num_iterations, verbose)

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
