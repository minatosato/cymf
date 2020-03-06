# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -std=c++11

import cython
import numpy as np
import pandas as pd
from scipy import sparse
from cython.parallel import prange
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set

from .math cimport UniformGenerator
from .model cimport BprModel
from .optimizer cimport Optimizer
from .optimizer cimport Sgd
from .optimizer cimport AdaGrad
from .optimizer cimport Adam

cdef extern from "util.h" namespace "cymf" nogil:
    cdef int threadid()
    cdef int cpucount()

class BPR(object):
    """
    Bayesian Personalized Ranking (BPR)
    https://arxiv.org/pdf/1205.2618.pdf
    
    Attributes:
        num_components (int): A dimensionality of latent vector
        learning_rate (double): A learning rate
        optimizer (str): Optimizers. e.g. 'adam', 'sgd'
        weight_decay (double): A coefficient of weight decay
        W (np.ndarray[double, ndim=2]): User latent vectors
        H (np.ndarray[double, ndim=2]): Item latent vectors
    """
    def __init__(self, int num_components = 20, double learning_rate = 0.001, str optimizer = "adam", double weight_decay = 0.01):
        """
        Args:
            num_components (int): A dimensionality of latent vector
            learning_rate (double): A learning rate
            optimizer (str): Optimizers. e.g. 'adam', 'sgd'
            weight_decay (double): A coefficient of weight decay
        """
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.W = None
        self.H = None

        if self.optimizer not in ("sgd", "adagrad", "adam"):
            raise Exception(f"{self.optimizer} is invalid.")

    def fit(self, X, int num_epochs = 10, int num_threads = 1, bool verbose = True):
        """
        Training BPR model with Gradient Descent.

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

        num_threads = num_threads if num_threads > 0 else cpucount()
        users, positives = utils.shuffle(*(X.nonzero()))

        return self._fit_bpr(users.astype(np.int32), 
                             positives.astype(np.int32),
                             X,
                             num_epochs,
                             self.learning_rate,
                             self.weight_decay,
                             num_threads,
                             verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_bpr(self,
                 int[:] users,
                 int[:] positives,
                 X,
                 int num_epochs, 
                 double learning_rate,
                 double weight_decay,
                 int num_threads,
                 bool verbose):

        cdef double[:,::1] W = self.W
        cdef double[:,::1] H = self.H
        cdef int iterations = num_epochs
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int U = X.shape[0]
        cdef int I = X.shape[1]
        cdef int u, l, iteration
        cdef double accum_loss

        cdef int user, positive, negative
        cdef vector[set[int]] user_positives = []
        cdef UniformGenerator gen = UniformGenerator(0, I, seed=1234)

        cdef Optimizer optimizer
        cdef BprModel bpr_model

        for u in range(U):
            user_positives.push_back({*X[u].nonzero()[1]})

        if self.optimizer == "adam":
            optimizer = Adam(self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = AdaGrad(self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = Sgd(self.learning_rate)

        optimizer.set_parameters(W, H)
        bpr_model= BprModel(W, H, optimizer, weight_decay, num_threads)

        with tqdm(total=iterations, leave=True, ncols=120, disable=not verbose) as progress:
            for iteration in range(iterations):
                accum_loss = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads, schedule="guided"):
                    user = users[l]
                    positive = positives[l]
                    negative = gen.generate()
                    if user_positives[user].find(negative) != user_positives[user].end():
                        continue
                    accum_loss += bpr_model.forward(user, positive, negative)
                    bpr_model.backward(user, positive, negative)

                accum_loss /= N

                progress.set_description(f"ITER={iteration+1}")
                progress.update(1)

