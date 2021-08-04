import glob
import os
import random
import tempfile

import numpy as np
from delfta.calculator import DelftaCalculator
from delfta.utils import TESTS_PATH
from openbabel.pybel import Outputfile, readfile
from sklearn.metrics import mean_absolute_error
import scipy

DELFTA_TO_DFT_KEYS = {
    "E_form": "DFT:FORMATION_ENERGY",
    "E_homo": "DFT:HOMO_ENERGY",
    "E_lumo": "DFT:LUMO_ENERGY",
    "E_gap": "DFT:HOMO_LUMO_GAP",
    "dipole": "DFT:DIPOLE",
    "charges": "DFT:MULLIKEN_CHARGES",
    # "wbo": "DFT:WIBERG_LOWDIN_BOND_ORDER", # TODO: add once implemented
}
CUTOFFS = {
    "E_form": (0.005, 0.05),
    "E_homo": (0.005, 0.005),
    "E_lumo": (0.005, 0.005),
    "E_gap": (0.005, 0.005),
    "dipole": (0.3, 0.4),
    "charges": (0.1, 0.1),
    # "wbo": (0.01, 0.01), # TODO: add once implemented
}


def test_invalid_mols_list():
    mol_files = random.sample(
        glob.glob(os.path.join(TESTS_PATH, "mols_working", "CHEMBL*.sdf")), 10
    )
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    invalid_mol_files = glob.glob(os.path.join(TESTS_PATH, "mols_invalid", "*.sdf"))
    invalid_mols = [next(readfile("sdf", mol_file)) for mol_file in invalid_mol_files]
    mols = mols + invalid_mols
    expected_nans = [True] * (len(mols) - len(invalid_mols)) + [False] * len(
        invalid_mols
    )
    tmp = list(zip(mols, expected_nans))
    random.shuffle(tmp)
    mols, expected_nans = zip(*tmp)
    calc_delta = DelftaCalculator(tasks=["all"], delta=True)
    predictions_delta = calc_delta.predict(
        list(mols), batch_size=random.randint(1, len(mols))
    )
    assert ~np.isnan(
        predictions_delta["E_form"][np.array(expected_nans)]
    ).all()  # valid molecules from SDF give result
    assert np.isnan(
        predictions_delta["E_form"][~np.array(expected_nans)]
    ).all()  # invalid_mols give NaN


def test_invalid_mols_generator():
    mol_files = random.sample(
        glob.glob(os.path.join(TESTS_PATH, "mols_working", "CHEMBL*.sdf")), 10
    )
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    invalid_mol_files = glob.glob(os.path.join(TESTS_PATH, "mols_invalid", "*.sdf"))
    invalid_mols = [next(readfile("sdf", mol_file)) for mol_file in invalid_mol_files]
    mols = mols + invalid_mols
    [mol.make2D() for mol in mols]
    expected_nans = [True] * (len(mols) - len(invalid_mols)) + [False] * len(
        invalid_mols
    )
    tmp = list(zip(mols, expected_nans))
    random.shuffle(tmp)
    mols, expected_nans = zip(*tmp)
    temp_dir = tempfile.TemporaryDirectory()
    test_sdf = os.path.join(temp_dir.name, "test_files.sdf")
    outfile = Outputfile("sdf", test_sdf)
    [outfile.write(mol) for mol in mols]
    calc_delta = DelftaCalculator(tasks=["all"], delta=True)
    predictions_delta = calc_delta.predict(
        test_sdf, batch_size=random.randint(1, len(mols))
    )
    assert ~np.isnan(
        predictions_delta["E_form"][np.array(expected_nans)]
    ).all()  # valid molecules from SDF give result
    assert np.isnan(
        predictions_delta["E_form"][~np.array(expected_nans)]
    ).all()  # invalid_mols give NaN


def test_3d_and_h_mols_list():
    filenames = [
        "no_3d_no_h.sdf",
        "no_3d_but_h.sdf",
        "yes_3d_but_no_h.sdf",
        "yes_3d_yes_h.sdf",
    ]
    mol_files = [
        os.path.join(TESTS_PATH, "mols_3d_h", filename) for filename in filenames
    ]
    assert len(mol_files) == 4

    # where do we expect a NaN answer because the calculator can't run with those options
    idxs_nan = [
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    force3ds = [True, True, False, False]
    addhs = [True, False, True, False]
    for force3d, addh, idx_nan in zip(force3ds, addhs, idxs_nan):
        mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
        calc_delta = DelftaCalculator(
            tasks=["all"], delta=True, force3d=force3d, addh=addh
        )
        predictions_delta = calc_delta.predict(
            mols, batch_size=random.randint(1, len(mols))
        )
        assert np.all(np.isnan(predictions_delta["E_form"][idx_nan]))


def test_3d_and_h_mols_generator():
    filenames = [
        "no_3d_no_h.sdf",
        "no_3d_but_h.sdf",
        "yes_3d_but_no_h.sdf",
        "yes_3d_yes_h.sdf",
    ]
    mol_files = [
        os.path.join(TESTS_PATH, "mols_3d_h", filename) for filename in filenames
    ]
    assert len(mol_files) == 4
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]
    temp_dir = tempfile.TemporaryDirectory()
    test_sdf = os.path.join(temp_dir.name, "test_files.sdf")
    outfile = Outputfile("sdf", test_sdf)
    [outfile.write(mol) for mol in mols]

    # where do we expect a NaN answer because the calculator can't run with those options
    idxs_nan = [
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    force3ds = [True, True, False, False]
    addhs = [True, False, True, False]
    for force3d, addh, idx_nan in zip(force3ds, addhs, idxs_nan):
        calc_delta = DelftaCalculator(
            tasks=["all"], delta=True, force3d=force3d, addh=addh
        )
        predictions_delta = calc_delta.predict(
            test_sdf, batch_size=random.randint(1, len(mols))
        )
        assert np.all(np.isnan(predictions_delta["E_form"][idx_nan]))


def test_calculator():
    mol_files = sorted(
        glob.glob(os.path.join(TESTS_PATH, "mols_working", "CHEMBL*.sdf"))
    )
    print(f"Located {len(mol_files)} sdf files for testing!")
    assert len(mol_files) == 100
    mols = [next(readfile("sdf", mol_file)) for mol_file in mol_files]

    calc_delta = DelftaCalculator(tasks="all", delta=True)
    predictions_delta = calc_delta.predict(mols)
    predictions_delta["charges"] = np.concatenate(predictions_delta["charges"])

    calc_direct = DelftaCalculator(tasks="all", delta=False)
    predictions_direct = calc_direct.predict(mols)
    predictions_direct["charges"] = np.concatenate(predictions_direct["charges"])

    # extract the ground truth from the QMugs SDFs
    dft_keys = [
        "DFT:FORMATION_ENERGY",
        "DFT:HOMO_ENERGY",
        "DFT:LUMO_ENERGY",
        "DFT:HOMO_LUMO_GAP",
        "DFT:DIPOLE",
        "DFT:MULLIKEN_CHARGES",
        # "DFT:WIBERG_LOWDIN_BOND_ORDER", # TODO: add once implemented
    ]
    dft_values = {}
    for dft_key in dft_keys:
        if dft_key == "DFT:DIPOLE":
            dft_values[dft_key] = [
                float(mol.data[dft_key].split("|")[-1]) for mol in mols
            ]
        elif (
            dft_key == "DFT:MULLIKEN_CHARGES"
            or dft_key == "DFT:WIBERG_LOWDIN_BOND_ORDER"
        ):
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


def test_xtb_opt():
    benzene_distorted = os.path.join(
        TESTS_PATH, "mols_xtb_opt", "benzene_distorted.xyz"
    )
    mol = next(readfile("xyz", benzene_distorted))
    old_coords = np.array([atom.coords for atom in mol.atoms])

    deltas = [True, False]
    for delta in deltas:
        calc_delta = DelftaCalculator(
            tasks=["E_form"], delta=delta, xtbopt=True, return_optmols=True
        )
        _, opt_mols = calc_delta.predict(mol)
        new_coords = np.array([atom.coords for atom in opt_mols[0].atoms])
        A = np.c_[new_coords[:, 0], new_coords[:, 1], np.ones(new_coords.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, new_coords[:, 2])  # coefficients
        Z = C[0] * new_coords[:, 0] + C[1] * new_coords[:, 1] + C[2]
        Z - new_coords[:, 2]
        assert ~np.all(np.equal(old_coords, new_coords))  # coordinates were modified
        assert np.all(
            np.isclose(0, Z - new_coords[:, 2], atol=1e-2)
        )  # molecule is planar


if __name__ == "__main__":
    test_calculator()
