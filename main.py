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

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=50)
parser.add_argument('--num_components', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

dataset: Dataset = MovieLens("ml-100k")

model = BPR(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
loss: float
with tqdm(total=args.iter, ncols=200) as progress:
    for i in range(args.iter):
        loss = model.fit_partial(dataset.train, num_iterations=1, num_threads=args.threads, verbose=False)
        description = f"ITER={i}, LOSS={loss:.4f}, NDCG@10={dcg_at_k(model, dataset.test, k=10).mean():.4f}, "
        description += f"Recall@10={recall_at_k(model, dataset.test, k=10).mean():.4f}"
        progress.set_description(description)
        progress.update(1)

# print(precision_at_k(model, dataset.test, k=10).mean())
# print(dcg_at_k(model, dataset.test, k=10).mean())

# model = WMF(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
# model.fit(dataset.train, num_iterations=3, num_threads=args.threads, verbose=True)

# print(precision_at_k(model, dataset.test, k=10).mean())
# print(dcg_at_k(model, dataset.test, k=10).mean())
