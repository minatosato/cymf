# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import os

cmpl_args = ['-Xpreprocessor', '-fopenmp', '-O3']
lnk_args = ['-lomp']

os.environ['CFLAGS'] = " ".join(cmpl_args + lnk_args)
os.environ['CXXFLAGS'] = " ".join(cmpl_args + lnk_args)

package_name: str = "fastmf"

setup(name=package_name, packages=[package_name], ext_modules=cythonize(["fastmf/*.pyx"]), include_dirs= [np.get_include()])

