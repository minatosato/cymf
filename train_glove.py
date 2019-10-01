
from utils import PTBDataset
from utils import to_glove_dataset

ptb_dataset = PTBDataset()

vocab_size = len(ptb_dataset.w2i)
embedding_size = 128
batch_size = 128
max_epoch = 100
window_size = 10


# TODO too slow. going to impl with c++
# central_train, context_train, target_train, X_train = to_glove_dataset(ptb_dataset.train_data, vocab_size=vocab_size, window_size=window_size)
central_valid, context_valid, target_valid, X_valid = to_glove_dataset(ptb_dataset.valid_data, vocab_size=vocab_size, window_size=window_size)

from glove import GloVe

model = GloVe(num_components=embedding_size, learning_rate=0.01, weight_decay=0.01)
model.fit(central_valid, context_valid, target_valid, X_valid, num_iterations=30, num_threads=4, verbose=True)