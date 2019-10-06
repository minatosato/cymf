# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3

import numpy as np
import cython
from cython cimport floating
from cython cimport integral
cimport numpy as np
from libcpp.vector cimport vector
import multiprocessing
from libcpp cimport bool
from sklearn import utils
import pandas as pd
from cython.parallel import prange
from threading import Thread
from cython.parallel import threadid
from tqdm import tqdm

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double pow(double x, double y) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline floating square(floating x) nogil:
    return x * x

class WMF(object):
    def __init__(self, int num_components,
                       floating learning_rate = 0.01,
                       floating weight_decay = 0.01,
                       floating weight = 5.0):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight = weight

    def fit(self, X, 
                  int num_iterations,
                  int num_threads,
                  bool verbose = False):
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components))
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components))

        # tmp = np.array([[u, i, X[u, i]] for u in range(X.shape[0]) for i in range(X.shape[1])])
        # users, items, ratings = utils.shuffle(tmp[:, 0], tmp[:, 1], tmp[:, 2])
        tmp = pd.DataFrame(X.todense()).stack().reset_index()
        tmp.columns = ("user", "item", "rating")
        users = tmp["user"].values
        items = tmp["item"].values
        ratings = tmp["rating"].values

        users, items, ratings = utils.shuffle(users, items, ratings)

        users = users.astype(np.int32)
        items = items.astype(np.int32)
        num_threads = min(num_threads, multiprocessing.cpu_count())
        fit_wmf(users, 
                items,
                ratings,
                self.W,
                self.H,
                num_iterations,
                self.learning_rate,
                self.weight_decay,
                self.weight,
                num_threads,
                verbose)

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_wmf(integral[:] users,
            integral[:] items,
            floating[:] ratings,
            floating[:,:] W, 
            floating[:,:] H, 
            int num_iterations, 
            floating learning_rate,
            floating weight_decay,
            floating weight,
            int num_threads,
            bool verbose):
    cdef int iterations = num_iterations
    cdef int N = users.shape[0]
    cdef int K = W.shape[1]
    cdef int u, i, j, k, l, iteration
    cdef floating[:] prediction = np.zeros(N)
    cdef floating[:] w_uk = np.zeros(N)
    cdef floating[:] l2_norm = np.zeros(N)
    cdef floating[:] diff = np.zeros(N)
    cdef floating[:] loss = np.zeros(N)
    cdef floating acc_loss
    
    cdef list description_list

    with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in prange(N, nogil=True, num_threads=num_threads):
                prediction[l] = 0.0
                l2_norm[l] = 0.0
                for k in range(K):
                    prediction[l] += W[users[l], k] * H[items[l], k]
                    l2_norm[l] += square(W[users[l], k])
                    l2_norm[l] += square(H[items[l], k])

                diff[l] = ratings[l] - prediction[l]
                if ratings[l] != 0.0:
                    diff[l] *= weight

                for k in range(K):
                    w_uk[l] = W[users[l], k]
                    W[users[l], k] += learning_rate * diff[l] * H[items[l], k]
                    H[items[l], k] += learning_rate * diff[l] * w_uk[l]
                
                loss[l] = square(diff[l])
                if ratings[l] != 0.0:
                    loss[l] = loss[l] * weight
                loss[l] += weight_decay * l2_norm[l]

            for l in range(N):                
                acc_loss += loss[l]

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"LOSS: {np.round(acc_loss/N, 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

