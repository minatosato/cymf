# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from scipy import sparse

from . import CooccurrrenceDataset
from ..glove import read_text

from pathlib import Path
from typing import Dict
from typing import Union

class Text8(CooccurrrenceDataset):
    def __init__(self, fname: str = "text8", min_count: int = 5, window_size = 10):
        super().__init__(fname, min_count, window_size)

        if not self.path.exists():
            import os
            os.system(f"wget http://mattmahoney.net/dc/{self.path.name}.zip")
            os.system(f"unzip {self.path.name}.zip")

        self.X, self.i2w = read_text(self.path.name, self.min_count, self.window_size)

    def vocab_size(self):
        return len(self.i2w)
