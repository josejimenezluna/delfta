FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
WORKDIR /usr/src/app

# install delfta
RUN git clone https://github.com/josejimenezluna/delfta.git

RUN cd delfta && make
RUN conda init
RUN echo 'conda activate delfta' >> ~/.bashrc
