SHELL := /bin/bash

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: install download

install:
	@ ./INSTALL.sh

download:
	@ ($(CONDA_ACTIVATE) delfta ; python setup.py install)
	@ ($(CONDA_ACTIVATE) delfta ; python delfta/download.py)
