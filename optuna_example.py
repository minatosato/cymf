# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from movielens import MovieLens
from dataset import Dataset

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=30)
parser.add_argument('--num_components', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--threads', type=int, default=4)

args = parser.parse_args()

dataset: Dataset = MovieLens("ml-100k")

from bpr import BPR
from metrics import auc
from metrics import precision_at_k

import optuna

def objective(trial: optuna.Trial):
    iterations = trial.suggest_int("iterations", 20, 100)
    model = BPR(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
    model.fit(dataset.train, num_iterations=iterations, num_threads=args.threads, verbose=False)
    return -precision_at_k(model, dataset.valid, k=5).mean()

study = optuna.create_study()
study.optimize(objective, n_trials=10)
print(study.best_params)

model = BPR(num_components=args.num_components, learning_rate=args.lr, weight_decay=args.weight_decay)
model.fit(dataset.train, num_iterations=study.best_params["iterations"], num_threads=args.threads, verbose=True)
print(precision_at_k(model, dataset.valid, k=5).mean())
