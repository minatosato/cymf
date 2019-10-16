
# fastmf
Cythonized fast implementation of matrix-factorization based algorithms.

### Implicit RecSys
- Bayesian Personlized Ranking [Steffen Rendle et al. 2009]
- Weighted Matrix Factorization [Yifan Hu et al. 2008]

### Word Embeddings
- GloVe [Jeffrey Pennington et al. 2014]

## Requiremts
- GCC 7.4.0
- libomp
- Python packages
    - see `requirements.txt`

## Installation
Ubuntu
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y g++-7
echo "export CXX='g++-7'" >> ~/.bashrc
echo "export CC='gcc-7'" >> ~/.bashrc
source ~/.bashrc
sudo apt install libomp-dev

git clone https://github.com/satopirka/fastmf
pip install -r requirements.txt
pip install .
```
