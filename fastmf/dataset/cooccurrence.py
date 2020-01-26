# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import numpy as np
from scipy import sparse

from pathlib import Path
from typing import Dict
from typing import Union

class CooccurrrenceDataset(object):
    path: Path
    min_count: int
    window_size: int
    i2w: Dict[int, str]
    X: Union[sparse.csr_matrix, sparse.csc_matrix]

    def __init__(self, fname: str, min_count: int = 5, window_size = 10):
        self.root: Path = Path.home().joinpath(".fastmf")
        self.root.mkdir(exist_ok=True)
        self.path: Path = self.root.joinpath(fname)
        self.min_count = min_count
        self.window_size = window_size

    def vocab_size(self):
        raise NotImplementedError()