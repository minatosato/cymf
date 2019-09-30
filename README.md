
```py
from movielens import MovieLens
from dataset import Dataset

dataset: Dataset = MovieLens()

from bpr import BPR
bpr = BPR(num_components=30,
          learning_rate=0.01,
          weight_decay=0.01)
bpr.fit(dataset.train,
        num_iterations=30,
        num_threads=4,
        verbose=True)
# ITER=300, LOSS: 0.2506: 100%|████████████████████| 300/300 [00:07<00:00, 42.80it/s]
```