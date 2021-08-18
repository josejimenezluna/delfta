"""
© 2021, ETH Zurich
"""

from openbabel.pybel import Molecule
from delfta.net_utils import QMUGS_ATOM_DICT

def _molcheck(mol):
    """Checks if `mol` is a valid molecule.

    Parameters
    ----------
    mol : any
        Any type, but should be openbabel.pybel.Molecule

    Returns
    -------
    bool
        `True` if `mol` is a valid molecule, `False` otherwise.
    """
    return (
        isinstance(mol, Molecule)
        and (mol.OBMol is not None)
        and (len(mol.atoms) > 0)
    )

def _3dcheck(mol, force3d):
    """Checks whether `mol` has 3d coordinates assigned. If
    `force3d=True` these will be computed for
    those lacking them using the MMFF94 force-field as
    available in pybel.

    Parameters
    ----------
    mol : pybel.Molecule
        An OEChem molecule object

    Returns
    -------
    bool
        `True` if `mol` has a 3d conformation, `False` otherwise.
    """
    if mol.dim != 3:
        if force3d:
            mol.make3D()
        return False
    return True

def _atomtypecheck(mol):
    """Checks whether the atom types in `mol` are supported
    by the QMugs database

    Parameters
    ----------
    mol : pybel.Molecule
        An OEChem molecule object

    Returns
    -------
    bool
        `True` if all atoms have valid atom types, `False` otherwise.
    """
    for atom in mol.atoms:
        if atom.atomicnum not in QMUGS_ATOM_DICT:
            return False
    return True

def _chargecheck(mol):
    """Checks whether the overall charge on `mol` is neutral.

    Parameters
    ----------
    mol : pybel.Molecule
        An OEChem molecule object

    Returns
    -------
    bool
        `True` is overall `mol` charge is 0, `False` otherwise.
    """
    if mol.charge != 0:
        return True
    else:
        return False

def _hydrogencheck(mol, addh):
    """Checks whether `mol` has assigned hydrogens. If `addh=True`
    these will be added if lacking.

    Parameters
    ----------
    mol : pybel.Molecule
        An OEChem molecule object

    Returns
    -------
    bool
        Whether `mol` has assigned hydrogens.
    """
    nh_mol = sum([True for atom in mol.atoms if atom.atomicnum == 1])
    mol_cpy = mol.clone
    mol_cpy.removeh()
    mol_cpy.addh()
    nh_cpy = sum([True for atom in mol_cpy.atoms if atom.atomicnum == 1])

    if nh_mol == nh_cpy:
        return True
    else:
        if addh:
            mol.addh()
        return False