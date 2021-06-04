import datetime
import json
import os
import shutil
import subprocess

import numpy as np
from openbabel import pybel

from delfta.utils import (
    ATOM_ENERGIES_XTB,
    ATOMNUM_TO_ELEM,
    AU_TO_DEBYE,
    EV_TO_HARTREE,
    XTB_BINARY,
    XTB_PATH,
)

XTB_INPUT_FILE = os.path.join(XTB_PATH, "xtb.inp")
TEMP_DIR = "/tmp/delfta_xtb/"
os.makedirs(TEMP_DIR, exist_ok=True)


def read_xtb_json(json_file, mol):
    """Reads JSON output file from xTB

    Args:
        json_file (str): path to output file
        mol (pybol Mol): molecule object, needed to compute atomic energy

    Returns:
        dict: dictionary of xTB properties
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    E_homo, E_lumo = get_homo_and_lumo_energies(data)
    atoms = [ATOMNUM_TO_ELEM[atom.atomicnum] for atom in mol.atoms]
    atomic_energy = sum([ATOM_ENERGIES_XTB[atom] for atom in atoms])
    props = {
        "E_form": data["total energy"] - atomic_energy,  # already in Hartree
        "E_homo": E_homo * EV_TO_HARTREE,
        "E_lumo": E_lumo * EV_TO_HARTREE,
        "E_gap": data["HOMO-LUMO gap/eV"] * EV_TO_HARTREE,
        "dipole": np.linalg.norm(data["dipole"]) * AU_TO_DEBYE,
        "charges": data["partial charges"],
    }
    assert np.isclose(
        props["E_lumo"] - props["E_homo"], props["E_gap"]
    )  # TODO @Jose: do such checks make sense, perhaps with a better error message?
    return props


def get_homo_and_lumo_energies(data):
    """Extracts HOMO and LUMO energies.

    Args:
        data (dict): dictionary from xTB JSON output

    Raises:
        ValueError: In case of unpaired electrons.

    Returns:
        tuple(float): HOMO/LUMO energies in eV
    """
    if data["number of unpaired electrons"] != 0:
        raise ValueError("Unpaired electrons are not supported.")
    num_occupied = (
        np.array(data["fractional occupation"]) != 0
    ).sum()  # number of occupied orbitals
    E_homo = data["orbital energies/eV"][num_occupied - 1]  # zero-indexing
    E_lumo = data["orbital energies/eV"][num_occupied]
    return E_homo, E_lumo


def run_xtb_calc(mol_file, opt=False):
    """Runs xtb single-point calculation with optional geometry optimization.

    Args:
        mol_file (str): path to mol file. Any format that Openbabel accepts.
        opt (bool, optional): Whether to optimize the geometry. Defaults to False.

    Raises:
        ValueError: If xTB calculation throws an error.

    Returns:
        dict: Molecular properties as computed by xTB (formation energy, HOMO/LUMO/gap energies, dipole, atomic charges)
    """
    xtb_command = "--opt" if opt else ""
    temp_logfile = os.path.join(TEMP_DIR, "xtb.log")
    json_file = os.path.join(TEMP_DIR, "xtbout.json")
    mol_file_wo_ext, ext = os.path.splitext(mol_file)

    if opt:
        temp_optfile = os.path.join(TEMP_DIR, "xtbopt" + ext)

    with open(temp_logfile, "w") as f:
        # run xTB
        completed_process = subprocess.run(
            [XTB_BINARY, mol_file, xtb_command, "--input", XTB_INPUT_FILE],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=TEMP_DIR,
        )
        if completed_process.returncode != 0:
            # copy logfile to input location if xTB failed
            t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            permanent_logfile = os.path.join(
                os.path.dirname(mol_file), f"xtb_error_{t}.log"
            )
            shutil.move(temp_logfile, permanent_logfile)
            raise ValueError(
                f"xTB calculation failed. See {permanent_logfile} for details."
            )  # TODO should we have more checks for e.g. unsuccessful geometry optimization?
    if opt:
        # copy the optimized file to the input location
        shutil.move(temp_optfile, mol_file_wo_ext + "_xtbopt" + ext)
    mol = next(
        pybel.readfile(ext.lstrip("."), mol_file)
    )  # TODO: pybel.ob.obErrorLog.SetOutputLevel(0)
    props = read_xtb_json(json_file, mol)
    return props


if __name__ == "__main__":
    import unittest
    from rdkit import Chem
    from delfta.utils import DATA_PATH

    class TestCase(unittest.TestCase):
        def test_compare_to_qmugs(self):
            mol_files = [
                "CHEMBL1342110_conf_02.sdf",
                "CHEMBL1346802_conf_01.sdf",
                "CHEMBL136619_conf_00.sdf",
                "CHEMBL2163771_conf_00.sdf",
                "CHEMBL251439_conf_02.sdf",
                "CHEMBL3108781_conf_01.sdf",
                "CHEMBL3287835_conf_00.sdf",
                "CHEMBL340588_conf_02.sdf",
                "CHEMBL3641659_conf_00.sdf",
                "CHEMBL3781981_conf_00.sdf",
            ]
            mol_files = [
                os.path.join(DATA_PATH, "test_data", elem) for elem in mol_files
            ]
            for mol_file in mol_files:
                print(f"Testing {mol_file}")
                props = run_xtb_calc(mol_file, opt=False)
                props_from_sdf = next(
                    Chem.SDMolSupplier(mol_file, removeHs=True)
                ).GetPropsAsDict()
                self.assertTrue(
                    np.isclose(
                        props["E_form"],
                        float(props_from_sdf["GFN2:FORMATION_ENERGY"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_homo"],
                        float(props_from_sdf["GFN2:HOMO_ENERGY"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_lumo"],
                        float(props_from_sdf["GFN2:LUMO_ENERGY"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_gap"],
                        float(props_from_sdf["GFN2:HOMO_LUMO_GAP"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["dipole"],
                        float(props_from_sdf["GFN2:DIPOLE"].split("|")[-1]),
                        atol=1e-2,
                    )
                )  # dipole is rounded
                charges_sdf = [
                    float(elem)
                    for elem in props_from_sdf["GFN2:MULLIKEN_CHARGES"].split("|")
                ]
                self.assertTrue(
                    np.all(np.isclose(props["charges"], charges_sdf, atol=1e-4))
                )

    unittest.main()
