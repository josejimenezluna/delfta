import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from openbabel.pybel import readfile, readstring
import numpy as np
from sklearn.metrics import mean_absolute_error

from delfta.calculator import DelftaCalculator
from delfta.utils import DATA_PATH

DELFTA_TO_DFT_KEYS = {
    "E_form": "DFT:FORMATION_ENERGY",
    "E_homo": "DFT:HOMO_ENERGY",
    "E_lumo": "DFT:LUMO_ENERGY",
    "E_gap": "DFT:HOMO_LUMO_GAP",
    "dipole": "DFT:DIPOLE",
    "charges": "DFT:MULLIKEN_CHARGES",
}
CUTOFFS = {
    "E_form": (0.005, 0.05),
    "E_homo": (0.005, 0.005),
    "E_lumo": (0.005, 0.005),
    "E_gap": (0.005, 0.005),
    "dipole": (0.3, 0.4),
    "charges": (0.1, 0.1)
}

def test_calculator():
    mol_files = sorted(glob.glob(os.path.join(DATA_PATH, "test_data", "CHEMBL*.sdf")))
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    calc_delta = DelftaCalculator(tasks=["all"], delta=True)
    predictions_delta = calc_delta.predict(mols)
    calc_direct = DelftaCalculator(tasks=["all"], delta=False)
    predictions_direct = calc_direct.predict(mols)
    predictions_delta["charges"] = np.concatenate(predictions_delta["charges"])
    predictions_direct["charges"] = np.concatenate(predictions_direct["charges"])

    dft_keys = [
        "DFT:FORMATION_ENERGY",
        "DFT:HOMO_ENERGY",
        "DFT:LUMO_ENERGY",
        "DFT:HOMO_LUMO_GAP",
        "DFT:DIPOLE",
        "DFT:MULLIKEN_CHARGES",
    ]
    dft_values = {}
    for dft_key in dft_keys:
        if dft_key == "DFT:DIPOLE":
            dft_values[dft_key] = [
                float(mol.data[dft_key].split("|")[-1]) for mol in mols
            ]
        elif dft_key == "DFT:MULLIKEN_CHARGES":
            dft_values[dft_key] = [
                float(elem) for mol in mols for elem in mol.data[dft_key].split("|")
            ]
        else:
            dft_values[dft_key] = [float(mol.data[dft_key]) for mol in mols]
    
    for key in DELFTA_TO_DFT_KEYS:
        pred_delta = predictions_delta[key]
        pred_direct = predictions_direct[key]
        dft_vals = np.array(dft_values[DELFTA_TO_DFT_KEYS[key]])
        assert mean_absolute_error(pred_delta, dft_vals) < CUTOFFS[key][0]
        assert mean_absolute_error(pred_direct, dft_vals) < CUTOFFS[key][1]

