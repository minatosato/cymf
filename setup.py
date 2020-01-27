# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
from pathlib import Path
from typing import Any, Match, Optional
import numpy as np
from Cython.Build import cythonize
from distutils.core import setup

cmpl_args = ['-Xpreprocessor', '-fopenmp', '-O3']
lnk_args = ['-lomp', '-lcblas']

os.environ['CFLAGS'] = " ".join(cmpl_args + lnk_args)
os.environ['CXXFLAGS'] = " ".join(cmpl_args + lnk_args)

package_name: str = "fastmf"

# export LDFLAGS="-L/usr/local/opt/openblas/lib"
# export CPPFLAGS="-I/usr/local/opt/openblas/include"

with Path(package_name).joinpath("__init__.py").open("r") as f:
    init_text = f.read()
    version: Optional[Match[Any]] = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text)
    license: Optional[Match[Any]] = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text)
    author: Optional[Match[Any]] = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text)
    author_email: Optional[Match[Any]] = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text)
    url: Optional[Match[Any]] = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'\"]', init_text)

if version is not None and license is not None and author is not None and author_email is not None and url is not None:
    setup(name=package_name,
        packages=[package_name, f"{package_name}.dataset"],
        version=version.group(1),
        author=author.group(1),
        author_email=author_email.group(1),
        ext_modules=cythonize(["fastmf/*.pyx"]),
        include_dirs= [np.get_include()])
