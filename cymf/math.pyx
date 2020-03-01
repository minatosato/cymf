#
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# cython: language_level=3
# distutils: language=c++

cdef class UniformGenerator(object):
    def __init__(self, long a, long b, unsigned int seed = 1234):
        self.rng = mt19937(seed)
        self.uniform = uniform_int_distribution[long](a, b-1)
    
    cdef inline long generate(self) nogil:
        return self.uniform(self.rng)