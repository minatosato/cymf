# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from gensim.models import KeyedVectors
from scipy import sparse
from pathlib import Path
from fastmf import GloVe
from fastmf.dataset import Text8
from fastmf.dataset import CooccurrrenceDataset

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--iter', type=int, default=15)
parser.add_argument('--num_components', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.75)
parser.add_argument('--x_max', type=float, default=10.0)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

print("text読み込み開始")
text8: CooccurrrenceDataset = Text8(min_count=args.min_count)
print("text読み込み完了")
embedding_size = 50

model = GloVe(num_components=args.num_components, learning_rate=args.lr, alpha=args.alpha, x_max=args.x_max)
model.fit(text8.X, num_iterations=args.iter, num_threads=args.threads, verbose=True)

output: Path = Path("./vectors.txt")
with output.open("w") as f:
    f.write(f"{model.W.shape[0]} {embedding_size}\n")
    for i in range(model.W.shape[0]):
        f.write(f"{text8.i2w[i]} " + " ".join(list(map(str, model.W[i]))) + "\n")

w2v = KeyedVectors.load_word2vec_format("./vectors.txt")
