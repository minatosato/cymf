# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from bpr import BPR
from wmf import WMF

from movielens import MovieLens
from dataset import Dataset

from metrics import auc
from metrics import precision_at_k
from metrics import recall_at_k
from metrics import dcg_at_k

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--num_components', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

dataset: Dataset = MovieLens("ml-100k")

model = BPR(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
history = model.fit(dataset.train, dataset.valid, dataset.test, num_iterations=args.iter, num_threads=args.threads, verbose=True)

df = pd.DataFrame(history)
df.columns = list(map(lambda x: x.decode("utf-8"), df.columns))
df.plot()
plt.grid()
plt.show()
