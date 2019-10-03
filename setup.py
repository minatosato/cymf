# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("bpr", ["bpr.pyx"], extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3'], extra_link_args=['-lomp'], language="c++"),
        Extension("wmf", ["wmf.pyx"], extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3'], extra_link_args=['-lomp'], language="c++"),
        Extension("glove", ["glove.pyx"], extra_compile_args=['-Xpreprocessor', '-fopenmp', '-O3'], extra_link_args=['-lomp'], language="c++")
        # -lcblas
    ],
    include_dirs= [np.get_include()]
)
