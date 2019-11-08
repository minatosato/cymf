FROM continuumio/miniconda3
RUN apt install -y build-essential git wget
RUN conda install -y -c conda-forge openmp
RUN cd ~ && wget https://raw.githubusercontent.com/satopirka/fastmf/master/requirements.txt && pip install -r requirements.txt
