# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import fastmf
import optuna
import numpy as np

from fastmf.dataset import ImplicitFeedbackDataset, MovieLens, YahooMusic

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

dataset: ImplicitFeedbackDataset = MovieLens("ml-100k")
valid_evaluator = fastmf.evaluator.Evaluator(dataset.valid.toarray(), metrics=["DCG"])
test_evaluator = fastmf.evaluator.Evaluator(dataset.test.toarray())

def bpr_objective(trial: optuna.Trial):
    iterations = trial.suggest_int("iterations", 10, 50)
    alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
    model = fastmf.BPR(num_components=args.num_components, learning_rate=alpha, weight_decay=weight_decay)
    model.fit(dataset.train, num_iterations=iterations, num_threads=args.threads, verbose=False)
    return valid_evaluator.evaluate(model.W@model.H.T)["DCG@5"]

def expomf_objective(trial: optuna.Trial):
    iterations = trial.suggest_int("iterations", 1, 10)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e1)
    model = fastmf.ExpoMF(num_components=args.num_components, weight_decay=weight_decay)
    model.fit(dataset.train, num_iterations=iterations, verbose=False)
    return valid_evaluator.evaluate(model.W@model.H.T)["DCG@5"]

def wmf_objective(trial: optuna.Trial):
    iterations = trial.suggest_int("iterations", 1, 10)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
    weight = trial.suggest_loguniform("weight", 1e0, 1e2)
    model = fastmf.WMF(num_components=args.num_components, weight_decay=weight_decay, weight=weight)
    model.fit(dataset.train, num_iterations=iterations, num_threads=args.threads, verbose=False)
    return valid_evaluator.evaluate(model.W@model.H.T)["DCG@5"]


summary = {}

study = optuna.create_study(direction="maximize")
study.optimize(bpr_objective, n_trials=100)
print(study.best_params)
model = fastmf.BPR(num_components=args.num_components, learning_rate=study.best_params["alpha"], weight_decay=study.best_params["weight_decay"])
model.fit(dataset.train, num_iterations=study.best_params["iterations"], num_threads=args.threads, verbose=True)
summary["BPR"] = test_evaluator.evaluate(model.W @ model.H.T)
print(summary["BPR"])

study = optuna.create_study(direction="maximize")
study.optimize(expomf_objective, n_trials=100)
print(study.best_params)
model = fastmf.ExpoMF(num_components=args.num_components, weight_decay=study.best_params["weight_decay"])
model.fit(dataset.train, num_iterations=study.best_params["iterations"], verbose=True)
summary["ExpoMF"] = test_evaluator.evaluate(model.W @ model.H.T)
print(summary["ExpoMF"])

study = optuna.create_study(direction="maximize")
study.optimize(wmf_objective, n_trials=100)
print(study.best_params)
model = fastmf.WMF(num_components=args.num_components, weight_decay=study.best_params["weight_decay"], weight=study.best_params["weight"])
model.fit(dataset.train, num_iterations=study.best_params["iterations"], num_threads=args.threads, verbose=False)
summary["WMF"] = test_evaluator.evaluate(model.W @ model.H.T)
print(summary["WMF"])

print(pd.DataFrame(summary))
