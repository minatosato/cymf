# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import cymf

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--clip_value', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

dataset = cymf.dataset.MovieLens("ml-100k")

valid_evaluator = cymf.evaluator.AverageOverAllEvaluator(dataset.valid, dataset.train, metrics=["DCG"], k=5)
test_evaluator = cymf.evaluator.AverageOverAllEvaluator(dataset.test, dataset.train, k=5)
model = cymf.RelMF(num_components=args.num_components, clip_value=0.6, weight_decay=args.weight_decay)
model.fit(dataset.train, num_epochs=args.max_epochs, num_threads=args.num_threads, valid_evaluator=valid_evaluator, early_stopping=True, verbose=True)
print(test_evaluator.evaluate(model.W, model.H))

