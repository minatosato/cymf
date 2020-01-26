# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import wget
import zipfile
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
            zip_path: Path = self.path.parent.joinpath(self.path.name + ".zip")
            if not zip_path.exists():
                wget.download(
                    f"http://mattmahoney.net/dc/{self.path.name}.zip",
                    out=str(zip_path)
                )

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.path.parent)

        self.X, self.i2w = read_text(str(self.path), self.min_count, self.window_size)

    def vocab_size(self):
        return len(self.i2w)
