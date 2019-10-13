# 
# Copyright (c) 2019 Minato Sato
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
    double fmin(double x, double y) nogil
    double sqrt(double x) nogil

cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef inline floating square(floating x) nogil:
    return x * x

cdef inline floating weight_func(floating x, floating x_max, floating alpha) nogil:
    return fmin(pow(x / x_max, alpha), 1.0)

cdef inline int imax(int a, int b) nogil:
    if (a > b):
        return a
    else:
        return b

cdef inline integral iabs(integral a) nogil:
    if (a < 0):
        return -a
    else:
        return a

class GloVe(object):
    def __init__(self, int num_components,
                       floating learning_rate = 0.01,
                       floating alpha = 0.75,
                       floating x_max = 10.0,
                       floating weight_decay = 0.01):
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.x_max = x_max
        self.weight_decay = weight_decay

    def fit(self, X,
                  int num_iterations,
                  int num_threads,
                  bool verbose = False):
                  
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0], self.num_components)) / self.num_components
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components
        _W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[1], self.num_components)) / self.num_components
        _bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components

        num_threads = min(num_threads, multiprocessing.cpu_count())
        central_words, context_words = X.nonzero()
        counts = X.data

        fit_glove(*utils.shuffle(central_words, context_words, counts),
                  self.W,
                  self.bias,
                  _W,
                  _bias,
                  num_iterations,
                  self.learning_rate,
                  self.x_max,
                  self.alpha,
                  self.weight_decay,
                  num_threads,
                  verbose)
        
        self.W = (self.W + _W) / 2.0


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_glove(integral[:] central_words,
              integral[:] context_words,
              floating[:] counts,
              floating[:,:] central_W,
              floating[:] central_bias,
              floating[:,:] context_W,
              floating[:] context_bias,
              int num_iterations, 
              floating learning_rate,
              floating x_max,
              floating alpha,
              floating weight_decay,
              int num_threads,
              bool verbose):
    cdef int iterations = num_iterations
    cdef int N = central_words.shape[0]
    cdef int N_W = central_W.shape[0]
    cdef int N_K = central_W.shape[1]
    cdef int u, i, j, k, l, iteration
    cdef floating[:] diff = np.zeros(N)
    cdef floating[:] loss = np.zeros(N)
    cdef floating[:] l2_norm = np.zeros(N)
    cdef floating[:] weight = np.zeros(N)
    cdef floating[:] grad = np.zeros(N)
    cdef floating[:] tmp = np.zeros(N)
    cdef floating acc_loss

    cdef floating[:,:] central_accum_gW = np.ones(shape=(N_W, N_K))
    cdef floating[:] central_accum_gbias = np.ones(shape=(N_W,))
    cdef floating[:,:] context_accum_gW = np.ones(shape=(N_W, N_K))
    cdef floating[:] context_accum_gbias = np.ones(shape=(N_W,))
    
    cdef list description_list

    with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in prange(N, nogil=True, num_threads=num_threads):
                diff[l] = 0.0
                l2_norm[l] = 0.0
                for k in range(N_K):
                    diff[l] += central_W[central_words[l], k] * context_W[context_words[l], k]
                diff[l] += central_bias[central_words[l]] + context_bias[context_words[l]] - log(counts[l])
                weight[l] = weight_func(counts[l], x_max, alpha)
                loss[l] = weight[l] * square(diff[l])

                for k in range(N_K):
                    tmp[l] = central_W[central_words[l], k]
                    grad[l] = weight[l] * diff[l] * context_W[context_words[l], k]
                    central_accum_gW[central_words[l], k] += square(grad[l])
                    central_W[central_words[l], k] -= learning_rate * grad[l] / sqrt(central_accum_gW[central_words[l], k])

                    grad[l] = weight[l] * diff[l] * tmp[l]
                    context_accum_gW[context_words[l], k] += square(grad[l])
                    context_W[context_words[l], k] -= learning_rate * grad[l] / sqrt(context_accum_gW[context_words[l], k])

                central_accum_gbias[central_words[l]] += square(weight[l] * diff[l])
                central_bias[central_words[l]] -= learning_rate * weight[l] * diff[l] / sqrt(central_accum_gbias[central_words[l]])
                context_accum_gbias[context_words[l]] += square(weight[l] * diff[l])
                context_bias[context_words[l]] -= learning_rate * weight[l] * diff[l] / sqrt(context_accum_gbias[context_words[l]])

            for l in range(N):
                acc_loss += loss[l]

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"LOSS: {np.round(acc_loss/N, 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)


@cython.boundscheck(False)
@cython.wraparound(False)
def read_text(str fname, int min_count = 5, int window_size = 10):
    cdef dict w2i, i2w, count
    cdef str raw
    cdef list words
    cdef list lines
    cdef vector[vector[int]] x = []
    cdef vector[int] tmp = []
    cdef int i, j, k, index, vocab_size
    cdef double[:,:] matrix
    with open(fname) as f:
        raw = f.read()
        words = raw.replace("\n", "<eos>").split(" ")
    count = dict(Counter(words))

    lines = raw.split("\n")

    w2i = {}
    i2w = {}
    for i in tqdm(range(len(lines)), ncols=100, leave=False):
        words = lines[i].split(" ")
        tmp = []
        for j in range(len(words)):
            if words[j] not in w2i and count[words[j]] >= min_count:
                index = len(w2i)
                w2i[words[j]] = index
                i2w[index] = words[j]
                tmp.push_back(index)
            elif count[words[j]] >= min_count:
                index = w2i[words[j]]
                tmp.push_back(index)
        x.push_back(tmp)

    vocab_size = len(w2i)
    matrix = np.zeros(shape=(vocab_size, vocab_size))
    for i in tqdm(range(len(x)), ncols=100, leave=False):
        for j in range(len(x[i])):
            for k in range(imax(0, j-window_size), j):
                matrix[x[i][j], x[i][k]] += 1.0 / iabs(j - k)

    return matrix, i2w