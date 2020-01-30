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
import multiprocessing
import numpy as np
import pandas as pd
from scipy import sparse
from cython.parallel import prange
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from libcpp cimport bool

from .model cimport BprModel
from .optimizer cimport Optimizer
from .optimizer cimport Sgd
from .optimizer cimport AdaGrad
from .optimizer cimport Adam

class BPR(object):
    """
    Bayesian Personalized Ranking (BPR)
    http://yifanhu.net/PUB/cf.pdf
    
    Attributes:
        num_components (int): Dimensionality of latent vector
        learning_rate (double): Leanring rate
        optimizer (str): Optimizers. e.g. 'adam', 'sgd'
        weight_decay (double): A coefficient of weight decay
        W (np.ndarray[double, ndim=2]): User latent vectors
        H (np.ndarray[double, ndim=2]): Item latent vectors
    """
    def __init__(self, int num_components, double learning_rate = 0.001, str optimizer = "adam", double weight_decay = 0.01):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.W = None
        self.H = None

        if self.optimizer not in ("sgd", "adagrad", "adam"):
            raise Exception(f"{self.optimizer} is invalid.")

    def fit(self, X, int num_iterations = 10, int num_threads = 8, bool verbose = False):
        """
        Training BPR model with Gradient Descent.
        https://arxiv.org/pdf/1205.2618.pdf

        Args:
            X: A user-item interaction matrix.
            num_iterations (int): The number of epochs.
            num_threads (int): The number of threads in HOGWILD! (http://i.stanford.edu/hazy/papers/hogwild-nips.pdf)
            verbose (bool): Whether to show the progress of training.
        """

        if isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            X = X.toarray()
        X = X.astype(np.float64)
        
        if self.W is None:
            self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        if self.H is None:
            self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

        num_threads = min(num_threads, multiprocessing.cpu_count())
        users, positives = utils.shuffle(*(X.nonzero()))

        return self._fit_bpr(users.astype(np.int32), 
                             positives.astype(np.int32),
                             X,
                             num_iterations,
                             self.learning_rate,
                             self.weight_decay,
                             num_threads,
                             verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_bpr(self,
                 int[:] users,
                 int[:] positives,
                 np.ndarray[double, ndim=2] X,
                 int num_iterations, 
                 double learning_rate,
                 double weight_decay,
                 int num_threads,
                 bool verbose):

        cdef double[:,::1] W = self.W
        cdef double[:,::1] H = self.H
        cdef int iterations = num_iterations
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, l, iteration
        cdef double[:] loss = np.zeros(N)
        cdef accum_loss
        cdef list description_list

        cdef int[:] negative_samples
        cdef int[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)

        cdef Optimizer optimizer
        if self.optimizer == "adam":
            optimizer = Adam(self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = AdaGrad(self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = Sgd(self.learning_rate)

        optimizer.set_parameters(W, H)

        cdef BprModel bpr_model = BprModel(W, H, optimizer, weight_decay, num_threads)

        for l in range(N):
            u = users[l]
            negative_samples = np.random.choice((X[u]-1).nonzero()[0], iterations).astype(np.int32)
            negatives[l][:] = negative_samples

        with tqdm(total=iterations, leave=True, ncols=120, disable=not verbose) as progress:
            for iteration in range(iterations):
                accum_loss = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = bpr_model.forward(users[l], positives[l], negatives[l, iteration])
                    bpr_model.backward(users[l], positives[l], negatives[l, iteration])

                for l in range(N):
                    accum_loss += loss[l]
                accum_loss /= N

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(accum_loss, 4):.4f}")

                progress.set_description(', '.join(description_list))
                progress.update(1)

