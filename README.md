# DelFTa: Open-source Δ-quantum machine learning
![](docs/delfta_overview.png)

[![delfta](https://github.com/josejimenezluna/delfta/actions/workflows/build.yml/badge.svg)](https://github.com/josejimenezluna/delfta/actions/workflows/build.yml)
![conda](https://anaconda.org/delfta/delfta/badges/installer/conda.svg)
[![Documentation Status](https://readthedocs.org/projects/delfta/badge/?version=latest)](https://delfta.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/josejimenezluna/delfta/branch/master/graph/badge.svg?token=kMkZiUi0DZ)](https://codecov.io/gh/josejimenezluna/delfta)
![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

## Installation

While the Linux/Windows installations fully support GPU-acceleration via cudatoolkit, only CPU inference is currently available under Mac OS. Additionally, only Python 3.7 and 3.8 are currently supported.

### Installation via conda

We recommend and support installation via the [conda](https://docs.conda.io/en/latest/miniconda.html) package manager, and that a fresh environment is created beforehand. 

```bash
conda install delfta -c delfta -c pytorch -c rusty1s -c conda-forge
```


### Installation via Docker

We also provide a CUDA-enabled Dockerfile for easier management. Build the container by

```bash
docker build -t delfta . 
```

Attach to the provided container with:

```bash
docker run -it delfta bash
```

## First run

DelFTa requires some additional files (_e.g._ trained models) before it can be used. Open a python CLI and execute the download script included in the package:

```python
import runpy
_ = runpy.run_module("delfta.download", run_name="__main__")
```

Alternatively, call the `_download_required` function in the `download` module:

```python
from delfta.download import _download_required
_download_required()
```

## Quick start

We interface with Pybel (OpenBabel). Most molecular file formats are supported (_e.g._ .sdf, .xyz).

```python
from openbabel.pybel import readstring
mol = readstring("smi", "CCO")

from delfta.calculator import DelftaCalculator
calc = DelftaCalculator()
preds = calc.predict(mol)

print(preds)
```


Further documentation on how to use the package is available under [ReadTheDocs](https://delfta.readthedocs.io/en/latest/).

## Tutorials

In-depth tutorials can be found in the `tutorials` subfolder. These include: 

- [delta_vs_direct.ipynb](tutorials/delta_vs_direct.ipynb): This showcases the basics of how to run the calculator, and compares results using direct- and Δ-learning models. 
- [calculator_options.ipynb](tutorials/calculator_options.ipynb): This dives into the different options you can initialize the calculator class with. 


## Citation

If you use this software or parts thereof, please consider citing the following BibTex entry:

```
@article{atz2021delfta,
  title={DelFTa: Open-source delta-quantum machine learning},
  author={Atz, K., and Isert, C., and B\"{o}cker, M., and Jiménez-Luna, J., and Schneider G.},
  journal={TBD},
  year={2021},
}
```
