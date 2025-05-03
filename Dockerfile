FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /usr/src/app

# Install micromamba
RUN apt-get update && apt-get install -y curl bzip2 git && \
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    rm -rf /var/lib/apt/lists/*

# Use only conda-forge
ENV MAMBA_NO_BANNER=1
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Create env
COPY environment.yml .
RUN micromamba create -y -n delfta -f environment.yml && \
    micromamba clean -a -y

# Add micromamba activation to .bashrc so it's available in interactive shells
RUN echo 'eval "$(/usr/local/bin/micromamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'micromamba activate delfta' >> /root/.bashrc

# Make sure PATH includes the micromamba bin directory
ENV PATH="/opt/conda/envs/delfta/bin:$PATH"

# Clone the delfta repository and install it
RUN git clone -b pytorch-update https://github.com/janash/delfta.git

RUN cd delfta && \
    pip install -e . && \
    rm -rf /root/.cache/pip

# Download module data
RUN cd delfta && \
    micromamba run -n delfta python -c "import runpy; _ = runpy.run_module('delfta.download', run_name='__main__')"

# Remove ENTRYPOINT to allow direct shell access with environment activated
CMD ["/bin/bash", "-l"]