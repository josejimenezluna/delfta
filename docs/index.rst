delfta
======

.. toctree::
   :maxdepth: 4

   delfta

Installation
------------

Pre-requisites
~~~~~~~~~~~~~~

While the Linux/Windows installations fully support GPU-acceleration
via cudatoolkit, only CPU inference is currently available under Mac OS.
Additionally, only Python 3.7 and 3.8 are currently supported.

Installation via conda
~~~~~~~~~~~~~~~~~~~~~~

We recommend and support installation via the
`conda <https://docs.conda.io/en/latest/miniconda.html>`__ package
manager, and that a fresh environment is created beforehand. 

.. code:: bash

   conda install delfta -c delfta -c pytorch -c rusty1s -c conda-forge


First run
~~~~~~~~~

DelFTa requires some additional files (e.g. trained models) before it
can be used. Open a python CLI and execute the download script included
in the package:

.. code:: bash
   
   python -c "import runpy; _ = runpy.run_module('delfta.download', run_name='__main__')"


Alternatively, call the ``_download_required`` function in the ``download`` module:


.. code-block:: python

   from delfta.download import _download_required
   _download_required()


Installation via Docker
~~~~~~~~~~~~~~~~~~~~~~~

A CUDA-enabled container can be pulled from `DockerHub <https://hub.docker.com/r/josejimenezluna/delfta>`__. 

We also provide a Dockerfile for manual builds:

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


Additional models
-----------------


Models trained on different training set sizes are available `here <https://polybox.ethz.ch/index.php/s/3mbn6iLwdleHSAh>`_.


Citation
--------


If you use this software or parts thereof, please consider citing the
following BibTex entry: ::

   @article{atz2021delfta,
      title={DelFTa: Open-source delta-quantum machine learning},
      author={Atz, K., and Isert, C., and B\"{o}cker, M., and Jiménez-Luna, J., and Schneider G.},
      journal={TBD},
      year={2021},
   }