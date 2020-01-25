FROM continuumio/miniconda3
RUN apt install -y build-essential git wget
RUN conda install -y -c conda-forge openmp
RUN apt install libopenblas-base
RUN apt install libopenblas-dev
RUN cd ~ && wget https://raw.githubusercontent.com/satopirka/fastmf/master/requirements.txt && pip install -r requirements.txt
RUN cd ~ && git clone https://raw.githubusercontent.com/satopirka/fastmf