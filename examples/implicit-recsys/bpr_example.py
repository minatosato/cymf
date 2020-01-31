# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fastmf
from fastmf.dataset import ImplicitFeedbackDataset, MovieLens, YahooMusic

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--num_threads', type=int, default=8)


args = parser.parse_args()

dataset: ImplicitFeedbackDataset = MovieLens("ml-100k")

Y_train = dataset.train.toarray()
Y_test = dataset.test.toarray()

evaluator = fastmf.evaluator.Evaluator(Y_test, unbiased=False)
model = fastmf.BPR(num_components=args.num_components, learning_rate=0.01, weight_decay=args.weight_decay)
for i in range(args.iter):
    model.fit(Y_train, num_iterations=1, num_threads=args.num_threads, verbose=False)
    print(evaluator.evaluate(model.W @ model.H.T))
