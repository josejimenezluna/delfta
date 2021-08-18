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
    mol = readstring("smi", "CCO")

    # run the prediction
    from delfta.calculator import DelftaCalculator
    calc = DelftaCalculator(tasks="all")
    preds = calc.predict(mol)

    print(preds)
    # >> {'E_homo': array([-0.35119605], dtype=float32), 'E_lumo': array([0.18145496], dtype=float32), 'E_gap': array([0.53317], dtype=float32), 'dipole': array([1.6470299], dtype=float32), 'E_form': array([-1.5151836], dtype=float32), 'charges': [array([-0.10071784,  0.13719846, -0.35453772,  0.0452125 ,  0.03470724, 0.02626099,  0.0383979 ,  0.01741201,  0.16891802])]}

You can also generate Openbabel molecules by *e.g.*, reading from a file to include atom coordinates. See the `Openbabel documentation <https://openbabel.org/docs/dev/UseTheLibrary/Python_Pybel.html>`__ for more information. 


Output
------
The following table shows the seven output endpoints for the delfta applications. Note in particular that energies are returned in Hartree, rather than eV.  

+-----------------------------------------------+-----------------+-------------------+
| **Property**                                  | **Key**         | **Unit**          |
+-----------------------------------------------+-----------------+-------------------+
| Formation energy                              | :code:`E_form`  | Hartree           |
+-----------------------------------------------+-----------------+-------------------+
| Energy of highest occupied molecular orbital  | :code:`E_homo`  | Hartree           |
+-----------------------------------------------+-----------------+-------------------+
| Energy of lowest unoccupied molecular orbital | :code:`E_lumo`  | Hartree           |
+-----------------------------------------------+-----------------+-------------------+
| HOMO-LUMO gap                                 | :code:`E_gap`   | Hartree           |
+-----------------------------------------------+-----------------+-------------------+
| Molecular dipole                              | :code:`dipole`  | Debye             |
+-----------------------------------------------+-----------------+-------------------+
| Mulliken partial charges                      | :code:`charges` | elementary charge |
+-----------------------------------------------+-----------------+-------------------+
| Wiberg bond orders                            | :code:`wbo`     | --                |
+-----------------------------------------------+-----------------+-------------------+


Tutorials
---------

In-depth tutorials can be found in the ``tutorials`` subfolder. These include: 

- ``delta_vs_direct.ipynb``: This showcases the basics of how to run the calculator, and compares results using direct- and :math:`\Delta`-learning models. 
- ``calculator_options.ipynb``: This dives into the different options you can initialize the calculator class with. 

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
