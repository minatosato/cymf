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

def update_ui(integral[:] users, integral[:] positives, integral[:] negatives, floating[:,:] W, floating[:,:] H, unsigned int num_iterations):
    cdef unsigned int iterations = num_iterations
    cdef unsigned int N = users.shape[0]
    cdef unsigned int K = W.shape[1]
    cdef unsigned int u, i, j, k, l, iteration
    cdef floating x_uij, w_uk
    cdef floating gradient_base
    cdef floating acc_loss, loss
    
    cdef list description_list

    with tqdm(total=iterations, leave=True, ncols=100) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in range(N):
                u = users[l]
                i = positives[l]
                j = negatives[l]

                x_uij = 0.0
                for k in range(K):
                    x_uij += W[u, k] * (H[i, k] - H[j, k])

                gradient_base = (1.0 / (1.0 + exp(x_uij)))

                for k in range(K):
                    w_uk = W[u, k]
                    W[u, k] += 0.01 * gradient_base * (H[i, k] - H[j, k])
                    H[i, k] += 0.01 * gradient_base * w_uk
                    H[j, k] += 0.01 * gradient_base * (-w_uk)

                loss = log(sigmoid(x_uij))
                acc_loss += loss

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"CURRENT MEAN LOSS: {np.round(acc_loss/N, 4):.4f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)



def train_bpr(integral[:] users, integral[:] positives, np.ndarray[floating, ndim=2] X, floating[:,:] W, floating[:,:] H, unsigned int num_iterations):
    # cdef integral[:] users = X.nonzero()[0]
    # cdef integral[:] positives = X.nonzero()[1]
    cdef unsigned int iterations = num_iterations
    cdef unsigned int N = users.shape[0]
    cdef unsigned int K = W.shape[1]
    cdef unsigned int u, i, j, k, l, iteration
    cdef floating x_uij, w_uk
    cdef floating gradient_base
    cdef floating acc_loss, loss
    
    cdef list description_list

    with tqdm(total=iterations, leave=True, ncols=100) as progress:
        for iteration in range(iterations):
            acc_loss = 0.0
            for l in range(N):
                u = users[l]
                i = positives[l]
                j = np.random.choice((X[u]-1).nonzero()[0])

                x_uij = 0.0
                for k in range(K):
                    x_uij += W[u, k] * (H[i, k] - H[j, k])

                gradient_base = (1.0 / (1.0 + exp(x_uij)))

                for k in range(K):
                    w_uk = W[u, k]
                    W[u, k] += 0.01 * gradient_base * (H[i, k] - H[j, k])
                    H[i, k] += 0.01 * gradient_base * w_uk
                    H[j, k] += 0.01 * gradient_base * (-w_uk)

                loss = log(sigmoid(x_uij))
                acc_loss += loss

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"CURRENT MEAN LOSS: {np.round(acc_loss/N, 4):.3f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

