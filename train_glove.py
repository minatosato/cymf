
from utils import PTBDataset
from utils import to_glove_dataset

ptb_dataset: PTBDataset = PTBDataset()

vocab_size = len(ptb_dataset.w2i)
embedding_size = 50
window_size = 10


# TODO too slow. going to impl with c++
central_train, context_train, target_train, X_train = to_glove_dataset(ptb_dataset.train_data, vocab_size=vocab_size, window_size=window_size)
central_valid, context_valid, target_valid, X_valid = to_glove_dataset(ptb_dataset.valid_data, vocab_size=vocab_size, window_size=window_size)

from glove import GloVe

model = GloVe(num_components=embedding_size, learning_rate=0.05, weight_decay=0.01)
model.fit(central_train, context_train, target_train, X_train, num_iterations=15, num_threads=8, verbose=True)

from pathlib import Path
output: Path = Path("./vectors.txt")
with output.open("w") as f:
    f.write(f"{model.W.shape[0]} {embedding_size}\n")
    for i in range(model.W.shape[0]):
        f.write(f"{ptb_dataset.i2w[i]} " + " ".join(list(map(str, model.W[i]))) + "\n")

import gensim
w2v = gensim.models.KeyedVectors.load_word2vec_format("./vectors.txt")
