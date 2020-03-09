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
    def __init__(self, lang: str = "en", min_count: int = 5, window_size = 10):
        fname: str
        if lang == "en":
            fname = "text8"
        elif lang == "ja":
            fname = "ja.text8"
        else:
            raise ValueError("An argument 'lang' must be 'en' or 'ja'.")

        super().__init__(fname, min_count, window_size)

        if not self.path.exists():
            zip_path: Path = self.path.parent.joinpath(self.path.name + ".zip")
            if not zip_path.exists():
                if lang == "en":
                    wget.download(
                        "http://mattmahoney.net/dc/text8.zip",
                        out=str(zip_path)
                    )
                elif lang == "ja":
                    wget.download(
                        "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/ja.text8.zip",
                        out=str(zip_path)
                    )

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.path.parent)

        self.X, self.i2w = read_text(str(self.path), self.min_count, self.window_size)

    def vocab_size(self):
        return len(self.i2w)
