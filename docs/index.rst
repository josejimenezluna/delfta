delfta
======

.. toctree::
   :maxdepth: 4

   delfta

Installation
------------

Pre-requisites
~~~~~~~~~~~~~~

Currently supported operating systems include both Linux and macOS (x86\_64
only). Basic build tools (*e.g*. ``make``) are currently required for
installation. Under Ubuntu, for instance, these are available under the
``build-essential`` package. The required package might be named
differently on different distributions.

While the Linux installation fully supports GPU-acceleration via
cudatoolkit, only CPU inference is currently available under macOS.

Installation via conda
~~~~~~~~~~~~~~~~~~~~~~

We recommend and support installation via the
`conda <https://docs.conda.io/en/latest/miniconda.html>`__ package
manager. First clone the repository to obtain the latest version or
download one of the provided stable releases:

.. code:: bash

    git clone https://github.com/josejimenezluna/delfta

Afterwards, move into the root repo directory and build the provided
Makefile via:

.. code:: bash

    cd delfta
    make

which will create a new conda environment named ``delfta`` and install
all the required dependencies as well as this package according to the
requirements of your host operating system. After the installation has
completed, you can now activate the environment and use the package via

.. code:: bash

    conda activate delfta

Installation via Docker
~~~~~~~~~~~~~~~~~~~~~~~

We also provide a CUDA-enabled Dockerfile for easier management. Build
the container with

.. code:: bash

    docker build -t delfta . 

Attach to the provided container with

.. code:: bash

    docker run -it delfta bash

Quick start
-----------
Here's a simple example of how to run a calculation for a single molecule created from a SMILES string:

.. code:: python

    # create an Openbabel molecule
    from openbabel.pybel import readstring
    mol = reastring("smi", "CCO")

    # run the prediction
    from delfta.calculator import DelftaCalculator
    calc = DelftaCalculator(tasks="all")
    preds = calc.predict(mol)

    print(preds)

You can also generate Openbabel molecules by *e.g.*, reading from a file to include atom coordinates. See the `Openbabel documentation <https://openbabel.org/docs/dev/UseTheLibrary/Python_Pybel.html>`__

Further documentation on how to use the package is available under
`ReadTheDocs <http://toinclude.html>`__.

Tutorials
---------

Several tutorials can be found under the ``tutorial`` subfolder. These
include:

TODO

Citation
--------

If you use this software or parts thereof, please consider citing the
following BibTex entry:

::

    @article{atz2021delfta,
        title={DelFTa: Open-source delta-quantum machine learning},
        author={Atz, K., and Isert, C., and B\"{o}cker, M., and Jiménez-Luna, J., and Schneider G.},
        journal={TBD},
        year={2021},
    }
