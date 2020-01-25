
# fastmf
[![CircleCI](https://circleci.com/gh/satopirka/fastmf.svg?style=svg)](https://circleci.com/gh/satopirka/fastmf)  
Cythonized fast implementation of matrix-factorization based algorithms.

### Implicit RecSys
- Bayesian Personlized Ranking (BPR) [Steffen Rendle et al. 2009]
- Weighted Matrix Factorization (WMF) [Yifan Hu et al. 2008]
- Exposure Matrix Factorization (ExpoMF) [Dawen Liang et al. 2016]

### Word Embeddings
- GloVe [Jeffrey Pennington et al. 2014]

## Requiremts
- GCC 7.4.0
- OpenMP
- OpenBLAS
- Python packages
    - see `requirements.txt`

## Installation
macOS
```
brew install libomp openblas

git clone https://github.com/satopirka/fastmf
cd fastmf
pip install -r requirements.txt
pip install .
```

Ubuntu
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y g++-7
echo "export CXX='g++-7'" >> ~/.bashrc
echo "export CC='gcc-7'" >> ~/.bashrc
source ~/.bashrc
sudo apt install libomp-dev
sudo apt-get install libopenblas-base
sudo apt-get install libopenblas-dev

git clone https://github.com/satopirka/fastmf
cd fastmf
pip install -r requirements.txt
pip install .
```
