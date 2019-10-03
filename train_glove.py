

from scipy import sparse
import nnabla.logger as L
from glove import GloVe
from glove import read_text

L.info("text読み込み開始")
X, i2w = read_text("ptb.train.txt", min_count=5)
L.info("text読み込み完了")

vocab_size = len(i2w)
embedding_size = 50

L.info("sparse matrix 変換開始")
sparse_X = sparse.csr_matrix(X)
L.info("sparse matrix 変換完了")

model = GloVe(num_components=embedding_size, learning_rate=0.05, weight_decay=0.01)
model.fit(sparse_X, num_iterations=15, num_threads=1, verbose=True)


from pathlib import Path
output: Path = Path("./vectors.txt")
with output.open("w") as f:
    f.write(f"{model.W.shape[0]} {embedding_size}\n")
    for i in range(model.W.shape[0]):
        f.write(f"{i2w[i]} " + " ".join(list(map(str, model.W[i]))) + "\n")

import gensim
w2v = gensim.models.KeyedVectors.load_word2vec_format("./vectors.txt")
