
from gensim.models import KeyedVectors
from scipy import sparse
from pathlib import Path
from fastmf import GloVe
from fastmf import read_text

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--iter', type=int, default=15)
parser.add_argument('--num_components', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.75)
parser.add_argument('--x_max', type=float, default=10.0)
parser.add_argument('--threads', type=int, default=8)

args = parser.parse_args()

print("text読み込み開始")
X, i2w = read_text("ptb.train.txt", min_count=args.min_count)
print("text読み込み完了")

vocab_size = len(i2w)
embedding_size = 50

print("sparse matrix 変換開始")
sparse_X = sparse.csr_matrix(X)
print("sparse matrix 変換完了")

model = GloVe(num_components=args.num_components, learning_rate=args.lr, alpha=args.alpha, x_max=args.x_max)
model.fit(sparse_X, num_iterations=args.iter, num_threads=args.threads, verbose=True)

output: Path = Path("./vectors.txt")
with output.open("w") as f:
    f.write(f"{model.W.shape[0]} {embedding_size}\n")
    for i in range(model.W.shape[0]):
        f.write(f"{i2w[i]} " + " ".join(list(map(str, model.W[i]))) + "\n")

w2v = KeyedVectors.load_word2vec_format("./vectors.txt")

