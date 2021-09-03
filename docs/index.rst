delfta
======

.. toctree::
   :maxdepth: 4

   delfta

Overview 
------------

The DelFTa application is an easy-to-use, open-source toolbox for 
predicting quantum-mechanical properties of drug-like molecules. 
Using either ∆-learning (with a GFN2-xTB baseline) or direct-learning (without a baseline), the application accurately approximates 
DFT reference values (*ω*\ B97X-D/def2-SVP). It employs state-of-the-art 
E(3)-equivariant graph neural networks trained on the QMugs dataset of 
quantum-mechanical properties, and can predict formation and orbital 
energies, dipoles, Mulliken partial charges and Wiberg bond orders. See the `paper <https://chemrxiv.org/engage/chemrxiv/article-details/612faa02abeb63218cc5f6f1>`__ for more details.

Installation
------------

Pre-requisites
~~~~~~~~~~~~~~

While the Linux (and Windows, through WSL) installations
fully support GPU-acceleration via cudatoolkit, only CPU
inference is currently available under Mac OS.
We currently support Python 3.7 and 3.8 builds.

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
    # >> {'E_form': array([-1.2982624], dtype=float32), 'charges': [array([-0.0972612 ,  0.1360395 , -0.35543405,  0.04585792,  0.03582753,
        0.02528055,  0.03844234,  0.01702337,  0.17144795])], 'wbo': [array([1.07290873, 1.1458627 , 0.95775243, 0.9563791 , 0.95719671,
       0.94609926, 0.94542666, 1.1332782 ])], 'E_homo': array([-0.34642667], dtype=float32), 'E_lumo': array([0.18278737], dtype=float32), 'E_gap': array([0.52899516], dtype=float32), 'dipole': array([1.9593117], dtype=float32)}

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
- ``training.ipynb``: A simple example of how networks can be trained. 


Additional models
-----------------


Models trained on different training set sizes are available `here <https://polybox.ethz.ch/index.php/s/3mbn6iLwdleHSAh>`_.


Citation
--------


If you use this software or parts thereof, please consider citing the
following BibTex entry: ::

   @article{atz2021delfta, 
      title={Open-source Δ-quantum machine learning for medicinal chemistry}, 
      DOI={10.33774/chemrxiv-2021-fz6v7}, 
      journal={ChemRxiv}, 
      publisher={Cambridge Open Engage}, 
      author={Atz, Kenneth and Isert, Clemens and Böcker, Markus N. A. and Jiménez-Luna, José and Schneider, Gisbert}, 
      year={2021}
   } 