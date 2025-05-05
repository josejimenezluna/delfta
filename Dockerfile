FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /usr/src/app

# Install dependencies and micromamba
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bzip2 git ca-certificates && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \ 
    tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    # Clean up apt caches
    apt-get clean && \
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
RUN git clone --depth 1 https://github.com/josejimenezluna/delfta.git

RUN cd delfta && \
    pip install -e . && \
    rm -rf /root/.cache/pip

# Download module data
RUN cd delfta && \
    micromamba run -n delfta python -c "import runpy; _ = runpy.run_module('delfta.download', run_name='__main__')"

CMD ["/bin/bash", "-l"]