# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from scipy.special import digamma
import cython
from cython cimport floating
from cython cimport integral
cimport numpy as np
from libcpp.vector cimport vector
import multiprocessing
from libcpp cimport bool
from sklearn import utils
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

cdef inline floating sqare(floating x) nogil:
    return x * x

class BPR(object):
    def __init__(self, unsigned int latent_dims,
                       unsigned int num_iterations,
                       floating learning_rate,
                       floating weight_decay,
                       unsigned int num_threads):
        self.latent_dims = latent_dims
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_threads = min(num_threads, multiprocessing.cpu_count())

    def fit(self, X, bool verbose = False):
        W = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[0], self.latent_dims))
        H = np.random.uniform(low=-0.1, high=0.1, size=(X.shape[1], self.latent_dims))

        users, positives = utils.shuffle(*(X.nonzero()))
        dense = np.array(X.todense())
        fit_bpr(users,
                positives,
                dense,
                W,
                H,
                self.num_iterations,
                self.learning_rate,
                self.weight_decay,
                self.num_threads,
                verbose)

        


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_bpr(integral[:] users,
            integral[:] positives,
            np.ndarray[floating, ndim=2] X, 
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
    cdef floating acc_loss, loss
    
    cdef list description_list

    cdef integral[:] negative_samples
    cdef integral[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)
    for l in range(N):
        u = users[l]
        negative_samples = np.random.choice((X[u]-1).nonzero()[0], iterations).astype(np.int32)
        negatives[l][:] = negative_samples

    with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in prange(N, nogil=True, num_threads=num_threads):
                x_uij[l] = 0.0
                l2_norm[l] = 0.0
                for k in range(K):
                    x_uij[l] += W[users[l], k] * (H[positives[l], k] - H[negatives[l][iteration], k])
                    l2_norm[l] += sqare(W[users[l], k])
                    l2_norm[l] += sqare(H[positives[l], k])
                    l2_norm[l] += sqare(H[negatives[l][iteration], k])

                gradient_base[l] = (1.0 / (1.0 + exp(x_uij[l])))

                for k in range(K):
                    w_uk[l] = W[users[l], k]
                    W[users[l], k] += learning_rate * (gradient_base[l] * (H[positives[l], k] - H[negatives[l][iteration], k]) - weight_decay * W[users[l], k])
                    H[positives[l], k] += learning_rate * (gradient_base[l] * w_uk[l] - weight_decay * H[positives[l], k])
                    H[negatives[l][iteration], k] += learning_rate * (gradient_base[l] * (-w_uk[l]) - weight_decay * H[negatives[l][iteration], k])

                loss = - log(sigmoid(x_uij[l])) + weight_decay * l2_norm[l]
                acc_loss += loss

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"LOSS: {np.round(acc_loss/N, 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

