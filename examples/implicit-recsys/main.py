# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import fastmf
from fastmf.dataset import ImplicitFeedBackDataset, MovieLens, YahooMusic

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--iter', type=int, default=50)
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--threads', type=int, default=1)

args = parser.parse_args()

dataset: ImplicitFeedBackDataset = MovieLens("ml-100k")

model = fastmf.BPR(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
history = model.fit(dataset.train, dataset.valid, dataset.test, num_iterations=args.iter, num_threads=args.threads, verbose=True)

history.plot()
plt.grid()
plt.show()
