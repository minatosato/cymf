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

# from math import exp
# from libc.math cimport exp as c_exp
# from libc.math cimport log as c_log
from cython.parallel import prange
from threading import Thread
from cython.parallel import threadid
from tqdm import tqdm

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil


srand48(1234)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline floating sigmoid(floating x) nogil:
    return 1.0 / (1.0 + exp(-x))

def train_bpr(integral[:] users, integral[:] positives, np.ndarray[floating, ndim=2] X, floating[:,:] W, floating[:,:] H, unsigned int num_iterations):
    # cdef integral[:] users = X.nonzero()[0]
    # cdef integral[:] positives = X.nonzero()[1]
    cdef unsigned int iterations = num_iterations
    cdef unsigned int N = users.shape[0]
    cdef unsigned int K = W.shape[1]
    cdef unsigned int u, i, j, k, l, iteration
    cdef floating[:] x_uij = np.zeros(N)
    cdef floating[:] w_uk = np.zeros(N)
    cdef floating gradient_base
    cdef floating acc_loss, loss
    
    cdef list description_list

    cdef integral[:] negative_samples
    cdef integral[:,:] negatives = np.zeros((N, iterations)).astype(np.int32)
    for l in tqdm(range(N), ncols=100):
        u = users[l]
        negative_samples = np.random.choice((X[u]-1).nonzero()[0], iterations).astype(np.int32)
        negatives[l][:] = negative_samples

    with tqdm(total=iterations, leave=True, ncols=100) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in prange(N, nogil=True):
                # users[l] = users[l]
                # positives[l] = positives[l]
                # negative_samples = negatives[l][iteration]
                # negatives[l][iteration] = negative_samples[iteration]

                x_uij[l] = 0.0
                for k in range(K):
                    x_uij[l] += W[users[l], k] * (H[positives[l], k] - H[negatives[l][iteration], k])

                gradient_base = (1.0 / (1.0 + exp(x_uij[l])))

                for k in range(K):
                    w_uk[l] = W[users[l], k]
                    W[users[l], k] += 0.01 * gradient_base * (H[positives[l], k] - H[negatives[l][iteration], k])
                    H[positives[l], k] += 0.01 * gradient_base * w_uk[l]
                    H[negatives[l][iteration], k] += 0.01 * gradient_base * (-w_uk[l])

                loss = log(sigmoid(x_uij[l]))
                acc_loss += loss

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"CURRENT MEAN LOSS: {np.round(acc_loss/N, 4):.3f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

