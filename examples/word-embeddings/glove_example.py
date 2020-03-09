# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from gensim.models import KeyedVectors
from pathlib import Path

import cymf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--language', type=str, default="ja")
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--num_components', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.75)
parser.add_argument('--x_max', type=float, default=10.0)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

print("loading text8...")
text8 = cymf.dataset.Text8(lang=args.language, min_count=args.min_count, window_size=args.window_size)

model = cymf.GloVe(num_components=args.num_components, learning_rate=args.lr, alpha=args.alpha, x_max=args.x_max)
model.fit(text8.X, num_epochs=args.num_epochs, num_threads=args.threads, verbose=True)
model.save_word2vec_format("./vectors.txt", text8.i2w)

w2v = KeyedVectors.load_word2vec_format("./vectors.txt")
