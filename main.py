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
parser.add_argument('--iter', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--threads', type=int, default=4)

args = parser.parse_args()

dataset: Dataset = MovieLens()

num_users: int = dataset.num_user
num_items: int = dataset.num_item
K = 30

from bpr import BPR

bpr = BPR(K, args.iter, args.lr, args.weight_decay, args.threads)
bpr.fit(dataset.train, verbose=True)
