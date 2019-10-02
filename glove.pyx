# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
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

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double pow(double x, double y) nogil
    double fmin(double x, double y) nogil
    double sqrt(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline floating square(floating x) nogil:
    return x * x

cdef inline floating weight_func(floating x, floating x_max) nogil:
    return fmin(pow(x / x_max, 0.75), 1.0)

class GloVe(object):
    def __init__(self, unsigned int num_components,
                       floating learning_rate = 0.01,
                       floating weight_decay = 0.01):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def fit(self, central_words, context_words, counts, X,
                  unsigned int num_iterations,
                  unsigned int num_threads,
                  bool verbose = False):
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0], self.num_components)) / self.num_components
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components
        self._W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[1], self.num_components)) / self.num_components
        self._bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components

        # central_words = X.nonzero()[0]
        # context_words = X.nonzero()[1]
        # counts = X.data
        # central_words, context_words, counts = utils.shuffle(central_words, context_words, counts)
        num_threads = min(num_threads, multiprocessing.cpu_count())

        fit_glove(*utils.shuffle(central_words, context_words, counts),
                  self.W,
                  self.bias,
                  self._W,
                  self._bias,
                  num_iterations,
                  self.learning_rate,
                  self.weight_decay,
                  num_threads,
                  verbose)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_glove(integral[:] central_words,
              integral[:] context_words,
              floating[:] counts,
              floating[:,:] central_W,
              floating[:] central_bias,
              floating[:,:] context_W,
              floating[:] context_bias,
              unsigned int num_iterations, 
              floating learning_rate,
              floating weight_decay,
              unsigned int num_threads,
              bool verbose):
    cdef unsigned int iterations = num_iterations
    cdef unsigned int N = central_words.shape[0]
    cdef unsigned int N_W = central_W.shape[0]
    cdef unsigned int N_K = central_W.shape[1]
    cdef unsigned int u, i, j, k, l, iteration
    cdef floating[:] diff = np.zeros(N)
    cdef floating[:] loss = np.zeros(N)
    cdef floating[:] l2_norm = np.zeros(N)
    cdef floating[:] weight = np.zeros(N)
    cdef floating[:] grad = np.zeros(N)
    cdef floating[:] tmp = np.zeros(N)
    cdef floating acc_loss

    cdef floating[:,:] central_accum_gW = np.zeros(shape=(N_W, N_K))
    cdef floating[:] central_accum_gbias = np.zeros(shape=(N_W,))
    cdef floating[:,:] context_accum_gW = np.zeros(shape=(N_W, N_K))
    cdef floating[:] context_accum_gbias = np.zeros(shape=(N_W,))
    
    cdef list description_list

    with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in prange(N, nogil=True, num_threads=num_threads):
                diff[l] = 0.0
                l2_norm[l] = 0.0
                for k in range(N_K):
                    diff[l] += central_W[central_words[l], k] * context_W[context_words[l], k]
                    # l2_norm[l] += square(central_W[central_words[l], k])
                    # l2_norm[l] += square(context_W[context_words[l], k])
                diff[l] += central_bias[central_words[l]] + context_bias[context_words[l]] - log(counts[l])
                weight[l] = weight_func(counts[l], 10)
                loss[l] = weight[l] * square(diff[l])

                for k in range(N_K):
                    tmp[l] = central_W[central_words[l], k]
                    grad[l] = weight[l] * diff[l] * context_W[context_words[l], k]
                    central_accum_gW[central_words[l], k] += square(grad[l])
                    central_W[central_words[l], k] -= learning_rate * grad[l] / sqrt(1.0 + central_accum_gW[central_words[l], k])

                    grad[l] = weight[l] * diff[l] * tmp[l]
                    context_accum_gW[context_words[l], k] += square(grad[l])
                    context_W[context_words[l], k] -= learning_rate * grad[l] / sqrt(1.0 + context_accum_gW[context_words[l], k])

                central_accum_gbias[central_words[l]] += square(weight[l] * diff[l])
                central_bias[central_words[l]] -= learning_rate * weight[l] * diff[l] / sqrt(1.0 + central_accum_gbias[central_words[l]])
                context_accum_gbias[context_words[l]] += square(weight[l] * diff[l])
                context_bias[context_words[l]] -= learning_rate * weight[l] * diff[l] / sqrt(1.0 + context_accum_gbias[context_words[l]])

            for l in range(N):
                acc_loss += loss[l]

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"LOSS: {np.round(acc_loss/N, 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def to_comatrix(vector[vector[int]] a):
#     cdef int sum = 0
#     cdef unsigned int i, j
#     for i in range(len(a)):
#         for j in range(len(a[i])):
#             sum += a[i][j]
#     return sum
