# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict
from typing import List

from functools import reduce
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.special import digamma

from movielens import MovieLens
from dataset import Dataset

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--ns_size', type=int, default=3)

args = parser.parse_args()

dataset: Dataset = MovieLens()

num_users: int = dataset.num_user
num_items: int = dataset.num_item
K = 30

W = np.random.uniform(low=-0.1, high=0.1, size=(num_users, K))
H = np.random.uniform(low=-0.1, high=0.1, size=(num_items, K))

from bpr import update_ui
from bpr import train_bpr

train_bpr(dataset.train.nonzero()[0], dataset.train.nonzero()[1], np.array(dataset.train.todense()), W, H, args.iter)

# update_ui(user_train, p_item_train, n_item_train, W, H, 100)
