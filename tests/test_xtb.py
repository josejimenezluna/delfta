import os
import glob

import numpy as np
from delfta.utils import TESTS_PATH
from delfta.xtb import run_xtb_calc
from openbabel.pybel import readfile
from tqdm import tqdm


def test_xtb_to_qmugs():
    mol_files = sorted(
        glob.glob(os.path.join(TESTS_PATH, "mols_working", "CHEMBL*.sdf"))
    )
    print(f"Located {len(mol_files)} sdf files for testing!")
    assert len(mol_files) == 100

    for sdf in tqdm(mol_files):
        mol = next(readfile("sdf", sdf))
        props = run_xtb_calc(mol, opt=False)
        assert np.isclose(
            props["E_form"], float(mol.data["GFN2:FORMATION_ENERGY"]), atol=1e-4,
        )

        assert np.isclose(
            props["E_homo"], float(mol.data["GFN2:HOMO_ENERGY"]), atol=1e-4,
        )

        assert np.isclose(
            props["E_lumo"], float(mol.data["GFN2:LUMO_ENERGY"]), atol=1e-4,
        )

        assert np.isclose(
            props["E_gap"], float(mol.data["GFN2:HOMO_LUMO_GAP"]), atol=1e-4,
        )
        assert np.isclose(
            props["dipole"], float(mol.data["GFN2:DIPOLE"].split("|")[-1]), atol=1e-2,
        )
        charges_sdf = [
            float(elem) for elem in mol.data["GFN2:MULLIKEN_CHARGES"].split("|")
        ]
        assert np.allclose(props["charges"], charges_sdf, atol=1e-4)
        wbo_sdf = [
            float(elem) for elem in mol.data["GFN2:WIBERG_BOND_ORDER"].split("|")
        ]
        assert np.allclose(props["wbo"], wbo_sdf, atol=1e-3)

