# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import cymf
import optuna
import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--alpha', type=float, default=1e-3)
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--num_threads', type=int, default=8)
parser.add_argument('--trials', type=int, default=10)

args = parser.parse_args()

dataset = cymf.dataset.MovieLens("ml-100k")

valid_evaluator = cymf.evaluator.AverageOverAllEvaluator(dataset.valid, dataset.train, k=5, metrics=["DCG"])
test_evaluator = cymf.evaluator.AverageOverAllEvaluator(dataset.test, dataset.train, k=5)

def bpr_objective(trial: optuna.Trial):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    model = cymf.BPR(num_components=args.num_components, learning_rate=args.alpha, weight_decay=weight_decay)
    model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
    return valid_evaluator.evaluate(model.W, model.H)["DCG@5"]

def expomf_objective(trial: optuna.Trial):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    model = cymf.ExpoMF(num_components=args.num_components, weight_decay=weight_decay)
    model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
    return valid_evaluator.evaluate(model.W, model.H)["DCG@5"]

def wmf_objective(trial: optuna.Trial):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    weight = trial.suggest_loguniform("weight", 1, 30)
    model = cymf.WMF(num_components=args.num_components, weight=weight, weight_decay=weight_decay)
    model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
    return valid_evaluator.evaluate(model.W, model.H)["DCG@5"]

def relmf_objective(trial: optuna.Trial):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
    clip_value = trial.suggest_uniform("clip_value", 0.1, 0.5)
    model = cymf.RelMF(num_components=args.num_components, learning_rate=args.alpha, weight_decay=weight_decay, clip_value=clip_value)
    model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
    return valid_evaluator.evaluate(model.W, model.H)["DCG@5"]

summary = {}

study = optuna.create_study(direction="maximize")
study.optimize(relmf_objective, n_trials=args.trials)
print(study.best_params)
result = []
model = cymf.RelMF(num_components=args.num_components, learning_rate=args.alpha, clip_value=study.best_params["clip_value"])
model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
for i in range(5):
    result.append(test_evaluator.evaluate(model.W, model.H, seed=i))
summary["RelMF"] = dict(pd.DataFrame(result).describe().loc[["mean", "std"]].T["mean"]) 
print(summary["RelMF"])


study = optuna.create_study(direction="maximize")
study.optimize(bpr_objective, n_trials=args.trials)
print(study.best_params)
result = []
model = cymf.BPR(num_components=args.num_components, learning_rate=args.alpha, weight_decay=study.best_params["weight_decay"])
model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
for i in range(5):
    result.append(test_evaluator.evaluate(model.W, model.H, seed=i))
summary["BPR"] = dict(pd.DataFrame(result).describe().loc[["mean", "std"]].T["mean"]) 
print(summary["BPR"])

study = optuna.create_study(direction="maximize")
study.optimize(expomf_objective, n_trials=args.trials)
print(study.best_params)
result = []
model = cymf.ExpoMF(num_components=args.num_components, weight_decay=study.best_params["weight_decay"])
model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
for i in range(5):
    result.append(test_evaluator.evaluate(model.W, model.H, seed=i))
summary["ExpoMF"] = dict(pd.DataFrame(result).describe().loc[["mean", "std"]].T["mean"]) 
print(summary["ExpoMF"])

study = optuna.create_study(direction="maximize")
study.optimize(wmf_objective, n_trials=args.trials)
print(study.best_params)
result = []
model = cymf.WMF(num_components=args.num_components, weight_decay=study.best_params["weight_decay"], weight=study.best_params["weight"])
model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
for i in range(5):
    result.append(test_evaluator.evaluate(model.W, model.H, seed=i))
summary["WMF"] = dict(pd.DataFrame(result).describe().loc[["mean", "std"]].T["mean"]) 
print(summary["WMF"])

print(pd.DataFrame(summary))
