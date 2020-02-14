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
from scipy import sparse
from collections import Counter
from cython.parallel import prange
from cython.parallel import threadid
from sklearn import utils
from tqdm import tqdm
from cython.operator import dereference, postincrement

cimport numpy as np
from cython cimport floating
from cython cimport integral
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from .model cimport GloVeModel
from .optimizer cimport GloVeAdaGrad

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
    """
    GloVe: Global Vectors for Word Representation
    https://nlp.stanford.edu/projects/glove/
    
    Attributes:
        num_components (int): A dimensionality of latent vector
        learning_rate (double): A learning rate used in AdaGrad
        alpha (double): See the paper.
        x_max (double): See the paper.
        W (np.ndarray[double, ndim=2]): Word vectors
    """
    def __init__(self, int num_components,
                       double learning_rate = 0.01,
                       double alpha = 0.75,
                       double x_max = 10.0):
        """
        Args:
            num_components (int): A dimensionality of latent vector
            learning_rate (double): A learning rate used in AdaGrad
            alpha (double): See the paper.
            x_max (double): See the paper.
        """
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.x_max = x_max
        self.W = None

    def fit(self, X, int num_iterations, int num_threads, bool verbose = False):
        """
        Training GloVe model with Gradient Descent.

        Args:
            X: A word-word cooccurrence matrix.
            num_iterations (int): A number of epochs.
            num_threads (int): A number of threads in HOGWILD! (http://i.stanford.edu/hazy/papers/hogwild-nips.pdf)
            verbose (bool): Whether to show the progress of training.
        """
        if X is None:
            raise ValueError()

        if not isinstance(X, (sparse.lil_matrix, sparse.csr_matrix, sparse.csc_matrix)):
            raise TypeError("X must be a type of scipy.sparse.*_matrix.")
                  
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0], self.num_components)) / self.num_components
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components
        _W = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[1], self.num_components)) / self.num_components
        _bias = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[0],)) / self.num_components

        num_threads = min(num_threads, multiprocessing.cpu_count())
        central_words, context_words = X.nonzero()
        counts = X.data

        self._fit_glove(*utils.shuffle(central_words, context_words, counts),
                        self.W,
                        self.bias,
                        _W,
                        _bias,
                        num_iterations,
                        self.learning_rate,
                        self.x_max,
                        self.alpha,
                        num_threads,
                        verbose)
        
        self.W = (self.W + _W) / 2.0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_glove(self,
                   integral[:] central_words,
                   integral[:] context_words,
                   double[:] counts,
                   double[:,:] central_W,
                   double[:] central_bias,
                   double[:,:] context_W,
                   double[:] context_bias,
                   int num_iterations, 
                   double learning_rate,
                   double x_max,
                   double alpha,
                   int num_threads,
                   bool verbose):
        cdef int iterations = num_iterations
        cdef int N = central_words.shape[0]
        cdef int N_K = central_W.shape[1]
        cdef double[:] loss = np.zeros(N)
        cdef int u, i, j, k, l, iteration

        cdef double accum_loss
        
        cdef list description_list

        cdef GloVeAdaGrad optimizer
        optimizer = GloVeAdaGrad(learning_rate)
        optimizer.set_parameters(central_W, context_W, central_bias, context_bias)

        cdef GloVeModel glove_model = GloVeModel(
            central_W, context_W, central_bias, context_bias, x_max, alpha, optimizer, num_threads)

        with tqdm(total=iterations, leave=True, ncols=100, disable=not verbose) as progress:
            for iteration in range(iterations):
                accum_loss = 0.0
                for l in prange(N, nogil=True, num_threads=num_threads):
                    loss[l] = glove_model.forward(central_words[l], context_words[l], counts[l])
                    glove_model.backward(central_words[l], context_words[l])

                for l in range(N):
                    accum_loss += loss[l]

                description_list = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"LOSS: {np.round(accum_loss/N, 4):.4f}")
                progress.set_description(', '.join(description_list))
                progress.update(1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
def read_text(str fname, int min_count = 5, int window_size = 10):
    cdef dict w2i, i2w, count
    cdef str raw
    cdef list words
    cdef list lines
    cdef vector[vector[int]] x = []
    cdef vector[int] tmp = []
    cdef int i, j, k, index
    cdef long vocab_size
    cdef double[:,:] matrix
    cdef unordered_map[long, double] sparse_matrix
    cdef unordered_map[long, double].iterator _iterator
    cdef long[:] row, col
    cdef double[:] data

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

    try:
        matrix = np.zeros(shape=(vocab_size, vocab_size))
        for i in tqdm(range(len(x)), ncols=100, leave=False):
            for j in range(len(x[i])):
                for k in range(imax(0, j-window_size), j):
                    matrix[x[i][j], x[i][k]] += 1.0 / iabs(j - k)
        from scipy import sparse
        ret = sparse.csr_matrix(matrix)
        return ret, i2w
    except MemoryError:
        for i in tqdm(range(len(x)), ncols=100, leave=False):
            for j in range(len(x[i])):
                for k in range(imax(0, j-window_size), j):
                    sparse_matrix[((<long> x[i][j]) + (<long> x[i][k]) * (<long>vocab_size))] += 1.0 / iabs(j - k)
                    
    
        from scipy import sparse
        row = np.zeros(sparse_matrix.size(), dtype=np.int64)
        col = np.zeros(sparse_matrix.size(), dtype=np.int64)
        data = np.zeros(sparse_matrix.size())

        i = 0
        _iterator = sparse_matrix.begin()
        while _iterator != sparse_matrix.end():
            row[i] = dereference(_iterator).first % vocab_size
            col[i] = dereference(_iterator).first / vocab_size
            data[i] = dereference(_iterator).second
            postincrement(_iterator)
            i += 1
        ret = sparse.csr_matrix((data, (row, col)), shape=(vocab_size, vocab_size))
        return ret, i2w
