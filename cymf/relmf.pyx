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
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from .model cimport RelMfModel
from .optimizer cimport Optimizer
from .optimizer cimport Sgd
from .optimizer cimport AdaGrad
from .optimizer cimport Adam

class RelMF(object):
    """
    Relevance Matrix Factorization (Rel-MF) Prototype
    https://arxiv.org/pdf/1909.03601.pdf
    """
    def __init__(self, int num_components = 20,
                       double clip_value = 0.1,
                       double learning_rate = 0.001,
                       str optimizer = "adam",
                       double weight_decay = 0.01):
        """
        Args:
            num_components (int): A dimensionality of latent vector
            clip_value (double): clip
            learning_rate (double): A learning rate
            optimizer (str): Optimizers. e.g. 'adam', 'sgd'
            weight_decay (double): A coefficient of weight decay
            weight (double): A weight for positive feedbacks.
        """
        self.num_components = num_components
        self.clip_value = clip_value
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.W = None
        self.H = None

        if self.optimizer not in ("sgd", "adagrad", "adam"):
            raise Exception(f"{self.optimizer} is invalid.")

    def fit(self, X, int num_epochs, int num_threads, bool verbose = False):
        """
        Training WMF model with Gradient Descent.
        Args:
            X: A user-item interaction matrix.
            num_epochs (int): A number of epochs.
            num_threads (int): A number of threads in HOGWILD! (http://i.stanford.edu/hazy/papers/hogwild-nips.pdf)
            verbose (bool): Whether to show the progress of training.
        """
        if X is None:
            raise ValueError()

        if isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            X = X.toarray()
        X = X.astype(np.float64)

        propensities = np.maximum(X.mean(axis=0), 1e-3)
        
        if self.W is None:
            np.random.seed(4321)
            self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        if self.H is None:
            self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

        tmp = pd.DataFrame(X).stack().reset_index()
        tmp.columns = ("user", "item", "rating")
        users = tmp["user"].values
        items = tmp["item"].values
        ratings = tmp["rating"].values

        users, items, ratings = utils.shuffle(users, items, ratings)

        users = users.astype(np.int32)
        items = items.astype(np.int32)
        num_threads = min(num_threads, multiprocessing.cpu_count())
        self._fit_wmf(users, 
                      items,
                      ratings,
                      propensities,
                      num_epochs,
                      num_threads,
                      verbose)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_wmf(self,
                 int[:] users,
                 int[:] items,
                 double[:] ratings,
                 double[:] propensities,
                 int num_epochs, 
                 int num_threads,
                 bool verbose):

        cdef double[:,:] W = self.W
        cdef double[:,:] H = self.H

        cdef int iterations = num_epochs
        cdef int N = users.shape[0]
        cdef int K = W.shape[1]
        cdef int u, i, j, k, l, iteration
        cdef double[:] prediction = np.zeros(N)
        cdef double[:] w_uk = np.zeros(N)
        cdef double[:] l2_norm = np.zeros(N)
        cdef double[:] diff = np.zeros(N)
        cdef double[:] loss = np.zeros(N)
        cdef double accum_loss

        cdef double M = self.clip_value
        
        cdef list description_list

        cdef Optimizer optimizer
        if self.optimizer == "adam":
            optimizer = Adam(self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = AdaGrad(self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = Sgd(self.learning_rate)

        optimizer.set_parameters(W, H)
        cdef RelMfModel relmf_model = RelMfModel(W, H, optimizer, self.weight_decay, num_threads)

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                accum_loss = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = relmf_model.forward(users[l], items[l], ratings[l], propensities[items[l]], M)
                    relmf_model.backward(users[l], items[l], ratings[l], propensities[items[l]], M)

                for l in range(N):                
                    accum_loss += loss[l]

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(accum_loss/N, 4):.4f}")
                progress.set_description(", ".join(description_list))
                progress.update(1)