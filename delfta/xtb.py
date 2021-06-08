import json
import os
import subprocess
import tempfile

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


def run_xtb_calc(mol, opt=False):
    """Runs xtb single-point calculation with optional geometry optimization.

    Args:
        mol (openbabel.pybel.Molecule): An OpenBabel molecule instance. 
        opt (bool, optional): Whether to optimize the geometry. Defaults to False.

    Raises:
        ValueError: If xTB calculation throws an error.

    Returns:
        dict: Molecular properties as computed by xTB (formation energy, HOMO/LUMO/gap energies,
              dipole, atomic charges)
    """
    xtb_command = "--opt" if opt else ""
    temp_dir = tempfile.TemporaryDirectory()
    logfile = os.path.join(temp_dir.name, "xtb.log")
    xtb_out = os.path.join(temp_dir.name, "xtbout.json")

    sdf_path = os.path.join(temp_dir.name, "mol.sdf")
    sdf = pybel.Outputfile("sdf", sdf_path)
    sdf.write(mol)
    sdf.close()

    with open(logfile, "w") as f:
        xtb_run = subprocess.run(
            [XTB_BINARY, sdf_path, xtb_command, "--input", XTB_INPUT_FILE],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=temp_dir.name,
        )

    if xtb_run.returncode != 0:
        error_out = os.path.join(temp_dir.name, "xtb_error.log")
        raise ValueError(f"xTB calculation failed. See {error_out} for details.")

    else:
        props = read_xtb_json(xtb_out, mol)
        temp_dir.cleanup()
    return props


if __name__ == "__main__":
    import unittest

    from delfta.utils import TESTS_PATH

    class TestCase(unittest.TestCase):
        def test_compare_to_qmugs(self):
            sdfs = [
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

            sdfs = [os.path.join(TESTS_PATH, sdf) for sdf in sdfs]

            for sdf in sdfs:
                mol = pybel.readfile("sdf", sdf).__next__()
                props = run_xtb_calc(mol, opt=False)

                self.assertTrue(
                    np.isclose(
                        props["E_form"],
                        float(mol.data["GFN2:FORMATION_ENERGY"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_homo"], float(mol.data["GFN2:HOMO_ENERGY"]), atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_lumo"], float(mol.data["GFN2:LUMO_ENERGY"]), atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["E_gap"],
                        float(mol.data["GFN2:HOMO_LUMO_GAP"]),
                        atol=1e-4,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        props["dipole"],
                        float(mol.data["GFN2:DIPOLE"].split("|")[-1]),
                        atol=1e-2,
                    )
                )  # dipole is rounded
                charges_sdf = [
                    float(elem) for elem in mol.data["GFN2:MULLIKEN_CHARGES"].split("|")
                ]
                self.assertTrue(np.allclose(props["charges"], charges_sdf, atol=1e-4))

    unittest.main()
