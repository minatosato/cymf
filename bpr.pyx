# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import cython
import multiprocessing
import numpy as np
from collections import Counter
from cython.parallel import prange
from cython.parallel import threadid
from sklearn import utils
from tqdm import tqdm

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

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

cdef inline string to_cpp_str(str x):
    return <string> x.encode("utf-8")

class BPR(object):
    def __init__(self, unsigned int num_components,
                       floating learning_rate = 0.01,
                       floating weight_decay = 0.01):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def fit(self, X, X_valid = None,
                  unsigned int num_iterations = 10,
                  unsigned int num_threads = 8,
                  bool verbose = False):

        self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
        self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components
        num_threads = min(num_threads, multiprocessing.cpu_count())

        users, positives = utils.shuffle(*(X.nonzero()))
        dense = np.array(X.todense())

        users_valid = None
        positives_valid = None
        dense_valid = None
        if X_valid is not None:
            users_valid, positives_valid = X_valid.nonzero()
            dense_valid = np.array(X_valid.todense())
            
        return fit_bpr(users, 
                       positives,
                       dense,
                       users_valid,
                       positives_valid,
                       dense_valid,
                       self.W,
                       self.H,
                       num_iterations,
                       self.learning_rate,
                       self.weight_decay,
                       num_threads,
                       verbose)

    # def fit_partial(self, X, 
    #                       unsigned int num_iterations,
    #                       unsigned int num_threads,
    #                       bool verbose = False):

    #     if not hasattr(self, "W"):
    #         self.W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.num_components)) / self.num_components
    #     if not hasattr(self, "H"):
    #         self.H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.num_components)) / self.num_components

    #     users, positives = X.nonzero()
    #     dense = np.array(X.todense())
    #     num_threads = min(num_threads, multiprocessing.cpu_count())
    #     return fit_bpr(users, 
    #                    positives,
    #                    dense,
    #                    self.W,
    #                    self.H,
    #                    num_iterations,
    #                    self.learning_rate,
    #                    self.weight_decay,
    #                    num_threads,
    #                    verbose)

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_bpr(integral[:] users,
            integral[:] positives,
            np.ndarray[floating, ndim=2] X,
            integral[:] users_valid,
            integral[:] positives_valid,
            np.ndarray[floating, ndim=2] X_valid,
            floating[:,:] W, 
            floating[:,:] H, 
            unsigned int num_iterations, 
            floating learning_rate,
            floating weight_decay,
            unsigned int num_threads,
            bool verbose):
    cdef unsigned int iterations = num_iterations
    cdef unsigned int N = users.shape[0]
    cdef unsigned int K = W.shape[1]
    cdef unsigned int u, i, j, k, l, iteration
    cdef floating[:] x_uij = np.zeros(N)
    cdef floating[:] w_uk = np.zeros(N)
    cdef floating[:] l2_norm = np.zeros(N)
    cdef floating[:] gradient_base = np.zeros(N)
    cdef floating[:] loss = np.zeros(N)

    cdef unordered_map[string, double] metrics
    
    cdef list description_list

    cdef integral[:] negative_samples
    cdef integral[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)
    cdef integral[:,:] negatives_valid = None

    cdef vector[unordered_map[string, double]] history = []

    for l in range(N):
        u = users[l]
        negative_samples = np.random.choice((X[u]-1).nonzero()[0], iterations).astype(np.int32)
        negatives[l][:] = negative_samples

    if X_valid is not None:
        negatives_valid = np.zeros((users_valid.shape[0], iterations)).astype(np.int32)
        for l in range(users_valid.shape[0]):
            u = users_valid[l]
            negative_samples = np.random.choice((X_valid[u]-1).nonzero()[0], iterations).astype(np.int32)
            negatives_valid[l][:] = negative_samples


    with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
        for iteration in range(iterations):
            metrics[to_cpp_str("loss")] = 0.0
            for l in prange(N, nogil=True, num_threads=num_threads):
                x_uij[l] = 0.0
                l2_norm[l] = 0.0
                for k in range(K):
                    x_uij[l] += W[users[l], k] * (H[positives[l], k] - H[negatives[l][iteration], k])
                    l2_norm[l] += square(W[users[l], k])
                    l2_norm[l] += square(H[positives[l], k])
                    l2_norm[l] += square(H[negatives[l][iteration], k])

                gradient_base[l] = (1.0 / (1.0 + exp(x_uij[l])))

                for k in range(K):
                    w_uk[l] = W[users[l], k]
                    W[users[l], k] += learning_rate * (gradient_base[l] * (H[positives[l], k] - H[negatives[l][iteration], k]) - weight_decay * W[users[l], k])
                    H[positives[l], k] += learning_rate * (gradient_base[l] * w_uk[l] - weight_decay * H[positives[l], k])
                    H[negatives[l][iteration], k] += learning_rate * (gradient_base[l] * (-w_uk[l]) - weight_decay * H[negatives[l][iteration], k])

                loss[l] = - log(sigmoid(x_uij[l])) + weight_decay * l2_norm[l]

            for l in range(N):
                metrics[to_cpp_str("loss")] += loss[l]
            metrics[to_cpp_str("loss")] /= N

            if X_valid is not None:
                metrics[to_cpp_str("val_loss")] = 0.0
                for l in prange(users_valid.shape[0], nogil=True, num_threads=num_threads):
                    x_uij[l] = 0.0
                    l2_norm[l] = 0.0
                    for k in range(K):
                        x_uij[l] += W[users_valid[l], k] * (H[positives_valid[l], k] - H[negatives_valid[l][iteration], k])
                        l2_norm[l] += square(W[users_valid[l], k])
                        l2_norm[l] += square(H[positives_valid[l], k])
                        l2_norm[l] += square(H[negatives_valid[l][iteration], k])

                    loss[l] = - log(sigmoid(x_uij[l])) + weight_decay * l2_norm[l]
                for l in range(users_valid.shape[0]):
                    metrics[to_cpp_str("val_loss")] += loss[l]
                metrics[to_cpp_str("val_loss")] /= users_valid.shape[0]

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"LOSS: {np.round(metrics[to_cpp_str('loss')], 4):.4f}")
            if X_valid is not None:
                description_list.append(f"VAL_LOSS: {np.round(metrics[to_cpp_str('val_loss')], 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

            history.push_back(metrics)
    return history

