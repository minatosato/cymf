FROM continuumio/miniconda3
RUN apt update
RUN apt install -y build-essential git wget
RUN conda install -y -c conda-forge openmp
RUN apt install -y libopenblas-base
RUN apt install -y libopenblas-dev
RUN apt install -y libatlas-base-dev
RUN conda install -y python=3.8
RUN pip install numpy scipy cython
RUN cd ~ && git clone https://github.com/satopirka/cymf
RUN pip install git+https://github.com/satopirka/cymf