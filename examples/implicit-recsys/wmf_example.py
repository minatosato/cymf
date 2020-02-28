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
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_components', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

dataset = cymf.dataset.MovieLens("ml-100k")
Y_train = dataset.train.toarray()
Y_test = dataset.test.toarray()

evaluator = cymf.evaluator.AverageOverAllEvaluator(Y_test, Y_train, k=5)
model = cymf.WMF(num_components=args.num_components, weight_decay=args.weight_decay)
#model.fit(Y_train, num_epochs=args.num_epochs, num_threads=args.num_threads, verbose=True)

for i in range(args.num_epochs):
    model.fit(Y_train, num_epochs=1, num_threads=args.num_threads, verbose=False)
    print(evaluator.evaluate(model.W @ model.H.T))
