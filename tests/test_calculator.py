import os
import glob
from openbabel.pybel import readfile, readstring, Molecule
import numpy as np
from sklearn.metrics import mean_absolute_error

from delfta.calculator import DelftaCalculator
from delfta.utils import TESTS_PATH

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
    "charges": (0.1, 0.1),
}

def test_invalid_mols(): 
    mol_files = sorted(glob.glob(os.path.join(TESTS_PATH, "CHEMBL*.sdf")))[:5]
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    invalid_mols = [
        Molecule(None),  # empty molecule
        readstring("smi", "[Xe]"),  # unsupported atom type
        readstring("smi", "CC[O-]"),  # charged
    ]
    mols = mols + invalid_mols
    calc_delta = DelftaCalculator(tasks=["all"], delta=True)
    predictions_delta = calc_delta.predict(mols)
    assert ~np.isnan(
        predictions_delta["E_form"][:-3]
    ).all()  # molecules from SDF give result
    assert np.isnan(predictions_delta["E_form"][-3:]).all()  # invalid_mols yield NaN


def test_3d_and_h_mols(): 
    filenames = ["no_3d_no_h.sdf", "no_3d_but_h.sdf", "yes_3d_but_no_h.sdf", "yes_3d_yes_h.sdf"]
    mol_files = [os.path.join(TESTS_PATH, "error_mols", filename) for filename in filenames]
    assert len(mol_files) == 4
    idxs_nan = [
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False]
    ]
    force3ds = [True, True, False, False]
    addhs = [True, False, True, False]
    for force3d, addh, idx_nan in zip(force3ds, addhs, idxs_nan): 
        mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
        calc_delta = DelftaCalculator(tasks=["all"], delta=True, force3d=force3d, addh=addh)
        predictions_delta = calc_delta.predict(mols)
        assert np.all(np.isnan(predictions_delta["E_form"][idx_nan]))

def test_calculator():
    mol_files = sorted(glob.glob(os.path.join(TESTS_PATH, "CHEMBL*.sdf")))
    print(f"Located {len(mol_files)} sdf files for testing!")
    assert len(mol_files) == 100

    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    calc_delta = DelftaCalculator(tasks=["all"], delta=True)
    predictions_delta = calc_delta.predict(mols)
    calc_direct = DelftaCalculator(tasks=["all"], delta=False)
    predictions_direct = calc_direct.predict(mols)
    predictions_delta["charges"] = np.concatenate(predictions_delta["charges"])
    predictions_direct["charges"] = np.concatenate(predictions_direct["charges"])

    # extract the ground truth from the QMugs SDFs
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

    # compare ground truth with prediction
    for key in DELFTA_TO_DFT_KEYS:
        pred_delta = predictions_delta[key]
        pred_direct = predictions_direct[key]
        dft_vals = np.array(dft_values[DELFTA_TO_DFT_KEYS[key]])
        assert mean_absolute_error(pred_delta, dft_vals) < CUTOFFS[key][0]
        assert mean_absolute_error(pred_direct, dft_vals) < CUTOFFS[key][1]


