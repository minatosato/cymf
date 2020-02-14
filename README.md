
# CyMF
![CyMF logo](logo.png)

||Ubuntu|macOS|
|-|-|-|
| Build status | [![CircleCI](https://circleci.com/gh/satopirka/cymf.svg?style=svg)](https://circleci.com/gh/satopirka/cymf) | [![Build Status](https://travis-ci.org/satopirka/cymf.svg?branch=master)](https://travis-ci.org/satopirka/cymf) |

Cython implementation of matrix-factorization based algorithms.

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
echo "export LDFLAGS='-L/usr/local/opt/openblas/lib'" >> ~/.bash_profile
echo "export CPPFLAGS='-I/usr/local/opt/openblas/include'" >> ~/.bash_profile
source ~/.bash_profile

pip install numpy scipy cython
pip install git+https://github.com/satopirka/cymf
```

Ubuntu
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y g++-7
echo "export CXX='g++-7'" >> ~/.bashrc
echo "export CC='gcc-7'" >> ~/.bashrc
source ~/.bashrc
sudo apt install libomp-dev libopenblas-base libopenblas-dev libatlas-base-dev

pip install numpy scipy cython
pip install git+https://github.com/satopirka/cymf
```
