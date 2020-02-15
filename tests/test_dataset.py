# 
# Copyright (c) 2020 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from cymf.dataset import ImplicitFeedbackDataset
from cymf.dataset import MovieLens
import pytest

movielens: ImplicitFeedbackDataset = MovieLens("ml-100k")

def test_movielens():
    assert movielens.train.shape == movielens.valid.shape == movielens.test.shape

def test_illegal_movielens_name():
    with pytest.raises(ValueError):
        MovieLens("ml-10b")
