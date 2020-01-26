# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from .implicit import ImplicitFeedbackDataset

class YahooMusic(ImplicitFeedbackDataset):
    def __init__(self, min_rating: float = 4.0, under_sampling: Optional[int] = None):
        dir_name: str = "yahoomusic"
        super().__init__(dir_name, min_rating)

        if not self.dir_path.exists():
            print("download R3 dataset from https://webscope.sandbox.yahoo.com/catalog.php?datatype=r ,")
            print(f"and put it on {self.dir_path.as_posix()}.")
            exit(1)

        self.df_train = pd.read_csv(self.dir_path.joinpath("ydata-ymusic-rating-study-v1_0-train.txt"),
                               sep="\t",
                               names=["user", "item", "rating"])
        self.df_train["user"] -= 1
        self.df_train["item"] -= 1
        self.df_train = self.df_train[self.df_train["rating"] >= min_rating]
        self.df_train["rating"] = 1.0

        self.df_test = pd.read_csv(self.dir_path.joinpath("ydata-ymusic-rating-study-v1_0-test.txt"),
                                   sep="\t",
                                   names=["user", "item", "rating"])
        self.df_test["user"] -= 1
        self.df_test["item"] -= 1
        self.df_test = self.df_test[self.df_test["rating"] >= min_rating]
        self.df_test["rating"] = 1.0

        self.num_user = max(self.df_train.user) + 1
        self.num_item = max(self.df_train.item) + 1

        self.df_train, self.df_valid = train_test_split(self.df_train, test_size=0.1, random_state=12345)

        self.train = self.to_matrix(self.df_train)
        self.valid = self.to_matrix(self.df_valid)
        self.test = self.to_matrix(self.df_test)

        # self.theta = np.array(self.train.sum(axis=0)).flatten()
        # self.theta /= self.theta.max()
        # self.theta[self.theta == 0] = 1 / self.train.shape[1]
        # self.theta = self.theta ** self.gamma

        self.train_size = self.train.nnz
        self.valid_size = self.valid.nnz
        self.test_size = self.test.nnz
