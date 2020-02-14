# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from typing import Tuple
from typing import Optional

class ImplicitFeedbackDataset(object):
    num_user: int
    num_item: int

    train_size: int
    valid_size: int
    test_size: int

    user_test: np.ndarray
    item_test: np.ndarray
    rating_test: np.ndarray

    train: sparse.lil_matrix
    valid: sparse.lil_matrix
    test: sparse.lil_matrix

    def __init__(self, dir_name: str, min_rating: float = 4.0, gamma: float = 0.2) -> None:
        self.root: Path = Path.home().joinpath(".cymf")
        self.root.mkdir(exist_ok=True)
        self.dir_path: Path = self.root.joinpath(dir_name)
        self.min_rating: float = min_rating

    def to_matrix(self, df: pd.DataFrame) -> sparse.lil_matrix:
        matrix = sparse.lil_matrix((self.num_user, self.num_item))
        for u, i, r in zip(df["user"].values, df["item"].values, df["rating"].values):
            matrix[u, i] = r
        return matrix
    
    def to_dataframe(self, matrix: sparse.lil_matrix) -> pd.DataFrame:
        df = pd.DataFrame(matrix.toarray()).stack().reset_index()
        df.columns = ("user", "item", "rating")
        df = df[df["rating"] >= 0]
        return df

    def split(self, df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return df.user.values, df.item.values, df.rating.values[:, None]
