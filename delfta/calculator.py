import collections
import os
import pickle
import types

import numpy as np
import openbabel
import torch
from torch_geometric.data.dataloader import DataLoader
from tqdm import tqdm

from delfta.download import get_model_weights
from delfta.net import EGNN
from delfta.net_utils import (
    MODEL_HPARAMS,
    MULTITASK_ENDPOINTS,
    QMUGS_ATOM_DICT,
    DeltaDataset,
)
from delfta.utils import ELEM_TO_ATOMNUM, LOGGER, MODEL_PATH
from delfta.xtb import run_xtb_calc

_ALLTASKS = ["E_form", "E_homo", "E_lumo", "E_gap", "dipole", "charges"]
PLACEHOLDER = float("NaN")


class DelftaCalculator:
    def __init__(
        self,
        tasks="all",
        delta=True,
        force3d=True,
        addh=True,
        xtbopt=False,
        verbose=True,
        progress=True,
        return_optmols=False,
    ) -> None:
        """Main calculator class for predicting DFT observables.

        Parameters
        ----------
        tasks : str, optional
            A list of tasks to predict. Available tasks include
            `[E_form, E_homo, E_lumo, E_gap, dipole, charges]`, by default "all".
        delta : bool, optional
            Whether to use delta-learning models, by default True
        force3d : bool, optional
            Whether to assign 3D coordinates to molecules lacking them, by default True
        addh : bool, optional
            Whether to add hydrogens to molecules lacking them, by default True
        xtbopt : bool, optional
            Whether to perform GFN2-xTB geometry optimization, by default False
        verbose : bool, optional
            Enables/disables stdout logger, by default True
        progress : bool, optional
            Enables/disables progress bars in prediction, by default True
        return_optmols: bool, optional
            Enables/disables returning the optimized molecules (use in combination with 
            xtbopt), by default False
        """
        if tasks == "all" or tasks == ["all"]:
            tasks = _ALLTASKS
        self.tasks = tasks
        self.delta = delta
        self.multitasks = [task for task in self.tasks if task in MULTITASK_ENDPOINTS]
        self.force3d = force3d
        self.addh = addh
        self.xtbopt = xtbopt
        self.verbose = verbose
        self.progress = progress
        self.return_optmols = return_optmols
        self.batch_mode = False

        if self.return_optmols and not self.xtbopt:
            raise ValueError("Only can use return_optmols in combination with xtbopt")

        with open(os.path.join(MODEL_PATH, "norm.pt"), "rb") as handle:
            self.norm = pickle.load(handle)

        self.models = []

        for task in tasks:
            if task in MULTITASK_ENDPOINTS:
                task_name = "multitask"

            elif task == "charges":
                task_name = "charges"

            elif task == "E_form":
                task_name = "single_energy"

            else:
                raise ValueError(f"Task name `{task}` not recognised")

            if self.delta:
                task_name += "_delta"
            else:
                task_name += "_direct"

            self.models.append(task_name)

        self.models = list(set(self.models))

    def _molcheck(self, mol):
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
            isinstance(mol, openbabel.pybel.Molecule)
            and (mol.OBMol is not None)
            and (len(mol.atoms) > 0)
        )

    def _3dcheck(self, mol):
        """Checks whether `mol` has 3d coordinates assigned. If
        `self.force3d=True` these will be computed for
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
            if self.force3d:
                mol.make3D()
            return False
        return True

    def _atomtypecheck(self, mol):
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

    def _chargecheck(self, mol):
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

    def _hydrogencheck(self, mol):
        """Checks whether `mol` has assigned hydrogens. If `self.addh=True`
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
            if self.addh:
                mol.addh()
            return False

    def _preprocess(self, mols, offset_idx=0):
        """Performs a series of preprocessing checks on a list of molecules `mols`,
        including 3d-conformation existence, validity of atom types, neutral charge
        and hydrogen addition.

        Parameters
        ----------
        mols: [pybel.Molecule]
            A list of OEChem molecule objects

        Returns
        -------
        ([pybel.Molecule], [int])
            A list of processed OEChem molecule objects; and indices of molecules that cannot be processed.
        offset_idx: int, optional
            By how much indices in the reported warnings should be offset. Helper for batch-wise processing.

        """
        if len(mols) == 0:
            raise ValueError("No molecules provided.")

        idx_non_valid_mols = set()
        idx_non_valid_atypes = set()
        idx_charged = set()
        idx_no3d = set()
        idx_noh = set()
        fatal = set()

        for idx, mol in enumerate(mols):
            valid_mol = self._molcheck(mol)
            if not valid_mol:
                idx_non_valid_mols.add(idx)
                fatal.add(idx)
                continue  # no need to check further

            is_atype_valid = self._atomtypecheck(mol)
            if not is_atype_valid:
                idx_non_valid_atypes.add(idx)
                fatal.add(idx)

            is_charged = self._chargecheck(mol)
            if is_charged:
                idx_charged.add(idx)
                fatal.add(idx)

            has_h = self._hydrogencheck(mol)
            if not has_h:
                idx_noh.add(idx)

            has_3d = self._3dcheck(mol)
            if not has_3d:
                idx_no3d.add(idx)

            if not has_3d and not has_h and not self.addh:
                fatal.add(idx)
                # cannot assign 3D geometry without assigning hydrogens

        good_mols = [mol for idx, mol in enumerate(mols) if idx not in fatal]
        self._log_status(
            len(mols),
            offset_idx,
            idx_non_valid_mols,
            idx_non_valid_atypes,
            idx_charged,
            idx_no3d,
            idx_noh,
        )
        if len(good_mols) == 0:
            raise ValueError("No valid molecules provided")
        return good_mols, list(fatal)

    def _log_status(
        self,
        num_mols,
        offset_idx,
        idx_non_valid_mols,
        idx_non_valid_atypes,
        idx_charged,
        idx_no3d,
        idx_noh,
    ):
        if idx_non_valid_mols:
            idx_non_valid_mols = {elem + offset_idx for elem in idx_non_valid_mols}
            LOGGER.warning(
                f"""Invalid molecules at position(s) {idx_non_valid_mols}. Skipping."""
            )
        if idx_non_valid_atypes:
            idx_non_valid_atypes = {elem + offset_idx for elem in idx_non_valid_atypes}
            LOGGER.warning(
                f"""Found non-supported atomic no. in molecules at position 
                    {idx_non_valid_atypes}. This application currently supports 
                    only the atom types used in the QMugs dataset, namely 
                    {list(ELEM_TO_ATOMNUM.keys())}. Skipping.
                """
            )
        if idx_charged:
            idx_charged = {elem + offset_idx for elem in idx_charged}
            LOGGER.warning(
                f"""Found molecules with a non-zero formal charge at 
                    position(s) {idx_charged}. This application currently does not support
                    prediction for charged molecules. Skipping.
                """
            )
        already_skipped = set.union(
            idx_non_valid_mols, idx_non_valid_atypes, idx_charged
        )
        has_3d = np.ones(num_mols, dtype=bool)
        if idx_no3d:
            has_3d[np.array(list(idx_no3d))] = False
        has_h = np.ones(num_mols, dtype=bool)
        if idx_noh:
            has_h[np.array(list(idx_noh))] = False

        # no hydrogens, no 3d
        if np.any(~has_3d & ~has_h):
            relevant_idx = np.where(~has_3d & ~has_h)[0].tolist()
            relevant_idx = [elem + offset_idx for elem in relevant_idx]
            relevant_idx = [
                elem for elem in relevant_idx if elem not in already_skipped
            ]
            if relevant_idx:
                if self.force3d and not self.addh:
                    LOGGER.warning(
                        f"""Molecules at position(s) {relevant_idx} have no 3D conformations 
                            and no hydrogens, but cannot `force3d` when `addh=False`. Skipping.
                        """
                    )
                elif self.force3d and self.addh and self.verbose:
                    LOGGER.info(
                        f"Assigned MMFF94 coordinates and added hydrogens to molecules at position(s) {relevant_idx}"
                    )
                elif not self.force3d and self.addh and self.verbose:
                    LOGGER.info(
                        f"""Added hydrogens to molecules at position(s) {relevant_idx}. They do not 
                        have 3D conformations available, but you didn't request to `force3d`."""
                    )

        # no hydrogens, but 3D
        if np.any(has_3d & ~has_h):
            relevant_idx = np.where(has_3d & ~has_h)[0].tolist()
            relevant_idx = [elem + offset_idx for elem in relevant_idx]
            relevant_idx = [
                elem for elem in relevant_idx if elem not in already_skipped
            ]
            if relevant_idx:
                if self.addh and self.verbose:
                    LOGGER.info(
                        f"""Added hydrogens to molecules at position(s) {relevant_idx}."""
                    )
                elif not self.addh and self.verbose:
                    LOGGER.info(
                        f"""Molecules at position(s) {relevant_idx} don't have hydrogens, but you didn't
                        request to `addh`."""
                    )

        # hydrogens, but no 3d
        if np.any(~has_3d & has_h):
            relevant_idx = np.where(~has_3d & has_h)[0].tolist()
            relevant_idx = [elem + offset_idx for elem in relevant_idx]
            relevant_idx = [
                elem for elem in relevant_idx if elem not in already_skipped
            ]
            if relevant_idx:
                if self.force3d and self.verbose:
                    LOGGER.info(
                        f"""Assigned MMFF94 coordinates to molecules at position(s) {relevant_idx}."""
                    )
                elif not self.force3d and self.verbose:
                    LOGGER.info(
                        f"""Molecules at position(s) {relevant_idx} don't have 3D conformations available,
                        but you didn't request to `force3d`."""
                    )

    def _get_preds(self, loader, model):
        """Returns predictions for the data contained in `loader` of a
        pyTorch `model`.


        Parameters
        ----------
        loader : delfta.net_utils.DeltaDataset
            A `delfta.net_utils.DeltaDataset` instance.
        model : delfta.net.EGNN
            A `delfta.net.EGNN` instance.

        Returns
        -------
        numpy.ndarray
            Model predictions.
        numpy.ndarray
            Graph-specific indexes for node-level predictions.
        """
        y_hats = []
        g_ptrs = []

        if self.progress and not self.batch_mode:
            progress = tqdm(total=len(loader.dataset))

        with torch.no_grad():
            for batch in loader:
                y_hats.append(model(batch).numpy())
                g_ptrs.append(batch.ptr.numpy())
                if self.progress and not self.batch_mode:
                    progress.update(n=batch.num_graphs)

        if self.progress and not self.batch_mode:
            progress.close()
        return y_hats, g_ptrs

    def _get_xtb_props(self, mols):
        """Runs the GFN2-xTB binary and returns observables

        Parameters
        ----------
        mols : [pybel.Molecule]
            A list of OEChem molecule instances.

        Returns
        -------
        dict
            A dictionary containing the requested properties for
            `mols`.
        list
            A list of indices with molecules for which the xtb calculation failed.
        list
            A list of optimized molecules (only returned if self.xtbopt==True)
        """
        xtb_props = collections.defaultdict(list)

        if self.verbose:
            LOGGER.info("Now running xTB...")

        if self.progress and not self.batch_mode:
            mols = tqdm(mols)

        if self.xtbopt:
            opt_mols = []

        fatal = []

        for i, mol in enumerate(mols):
            try:
                if self.xtbopt:
                    xtb_out, opt_mol = run_xtb_calc(mol, opt=True, return_optmol=True)
                    opt_mols.append(opt_mol)  # transfer the optimized geometry
                else:
                    xtb_out = run_xtb_calc(mol, opt=False, return_optmol=False)
                for prop, val in xtb_out.items():
                    xtb_props[prop].append(val)
            except ValueError as xtb_error:
                fatal.append(i)
                if self.xtbopt:
                    opt_mols.append(PLACEHOLDER)
                LOGGER.warning(xtb_error.args[0])
        return (xtb_props, fatal, opt_mols) if self.xtbopt else (xtb_props, fatal)

    def _inv_scale(self, preds, norm_dict):
        """Inverse min-max scaling transformation

        Parameters
        ----------
        preds : np.ndarray
            Normalized predictions
        norm_dict : dict
            A dictionary containing scale and location values for
            inverse normalization.

        Returns
        -------
        numpy.ndarray
            Unnormalized predictions in their original scale
            and location.
        """
        return preds * norm_dict["scale"] + norm_dict["location"]

    def _predict_batch(self, generator, batch_size):
        """Utility method for prediction using OEChem generators
        (e.g. those used for reading sdf or xyz files)

        Parameters
        ----------
        generator : pybel.filereader
            A pybel.filereader instance
        batch_size : int
            Batch size used for prediction. Defaults to the same one
            used under `self.predict`.

        Returns
        -------
        dict
            Requested DFT-predicted properties.
        """
        preds_batch = []
        if self.xtbopt:
            opt_mols_batch = []
        done_flag = False
        total_done_so_far = 0
        progress = tqdm()
        while not done_flag:
            mols = []
            done_so_far = 0
            for _ in range(batch_size):
                try:
                    mol = next(generator)
                    mols.append(mol)
                    done_so_far += 1
                except StopIteration:
                    done_flag = True
                    break
                except:
                    mols.append(PLACEHOLDER)  # invalid mol
                    done_so_far += 1

            if self.progress:
                progress.update(n=done_so_far)

            if self.xtbopt:
                preds_this_batch, opt_mols_this_batch = self.predict(
                    mols, batch_size, offset_idx=total_done_so_far
                )
                opt_mols_batch.extend(opt_mols_this_batch)
            else:
                preds_this_batch = self.predict(
                    mols, batch_size, offset_idx=total_done_so_far
                )

            preds_batch.append(preds_this_batch)
            total_done_so_far += done_so_far
        progress.close()

        pred_keys = preds_batch[0].keys()
        preds = collections.defaultdict(list)
        for pred_k in pred_keys:
            for batch in preds_batch:
                if pred_k == "charges":
                    preds[pred_k].extend(batch[pred_k])
                else:
                    preds[pred_k].extend(batch[pred_k].tolist())
            if pred_k != "charges":
                preds[pred_k] = np.array(preds[pred_k], dtype=np.float32)

        return (dict(preds), opt_mols_batch) if self.xtbopt else dict(preds)

    def predict(self, input_, batch_size=32, offset_idx=0):
        """Main prediction method for DFT observables.

        Parameters
        ----------
        input_ : None
            Either a single or a list of OEChem Molecule instances or a pybel filereader generator instance.
        batch_size : int, optional
            Batch size used for prediction, by default 32
        offset_idx: int, optional
            By how much indices in the reported warnings should be offset. Helper for batch-wise processing. Don't set manually.

        Returns
        -------
        dict
            Requested DFT-predicted properties.
        list
            A list of optimized molecules (only returned if self.xtbopt==True and self.return_optmol)
        """

        fatal_xtb, fatal = [], []

        if isinstance(input_, openbabel.pybel.Molecule):
            return self.predict([input_])

        elif isinstance(input_, list):
            mols, fatal = self._preprocess(input_, offset_idx=offset_idx)

        elif isinstance(input_, types.GeneratorType):
            if self.return_optmols:
                LOGGER.warning("Using return_optmol flag with a generator. This might cause memory issues if the input file is large.")
            self.batch_mode = True
            return self._predict_batch(input_, batch_size)

        elif isinstance(input_, str):
            ext = os.path.splitext(input_)[-1].lstrip(".")
            return self.predict(openbabel.pybel.readfile(ext, input_))

        else:
            raise ValueError(
                f"Invalid input. Expected OEChem molecule, list or generator, but got {type(input_)}."
            )

        if self.xtbopt:
            xtb_props, fatal_xtb, mols = self._get_xtb_props(mols)
        elif not self.xtbopt and self.delta:
            xtb_props, fatal_xtb = self._get_xtb_props(mols)

        mols = [mol for i, mol in enumerate(mols) if i not in fatal_xtb]
        data = DeltaDataset(mols)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        preds = {}

        for _, model_name in enumerate(self.models):
            if self.verbose:
                LOGGER.info(f"Now running network for model {model_name}...")
            model_param = MODEL_HPARAMS[model_name]
            model = EGNN(
                n_outputs=model_param.n_outputs, global_prop=model_param.global_prop
            ).eval()
            weights = get_model_weights(model_name)
            model.load_state_dict(weights)
            y_hat, g_ptr = self._get_preds(loader, model)

            if "charges" in model_name:
                atom_y_hats = []
                for batch_idx, batch_ptr in enumerate(g_ptr):
                    atom_y_hats.extend(
                        [
                            y_hat[batch_idx][batch_ptr[idx] : batch_ptr[idx + 1]]
                            for idx in range(len(batch_ptr) - 1)
                        ]
                    )
                preds[model_name] = atom_y_hats
            else:
                y_hat = np.vstack(y_hat)

                if "multitask" in model_name:
                    if "direct" in model_name:
                        y_hat = self._inv_scale(y_hat, self.norm["direct"])
                    else:
                        y_hat = self._inv_scale(y_hat, self.norm["delta"])

                preds[model_name] = y_hat

        preds_filtered = {}

        for model_name in preds.keys():
            mname = model_name.rsplit("_", maxsplit=1)[0]
            if mname == "single_energy":
                preds_filtered["E_form"] = preds[model_name].squeeze()
            elif mname == "multitask":
                for task in self.multitasks:
                    preds_filtered[task] = preds[model_name][
                        :, MULTITASK_ENDPOINTS[task]
                    ]

            elif mname == "charges":
                preds_filtered["charges"] = preds[model_name]

        if self.delta:
            for prop, delta_arr in preds_filtered.items():
                if prop == "charges":
                    preds_filtered[prop] = [
                        d_arr + np.array(xtb_arr)
                        for d_arr, xtb_arr in zip(delta_arr, xtb_props[prop])
                    ]
                else:
                    preds_filtered[prop] = delta_arr + np.array(
                        xtb_props[prop], dtype=np.float32
                    )

        if fatal_xtb:  # insert placeholder values where xtb errors occurred
            preds_filtered = self._insert_placeholders(
                preds_filtered, len(input_) - len(fatal), fatal_xtb
            )

        if fatal:  # insert placeholder values where other errors occurred
            preds_filtered = self._insert_placeholders(
                preds_filtered, len(input_), fatal
            )

        return (preds_filtered, mols) if self.return_optmols else preds_filtered

    def _insert_placeholders(self, preds, len_input, fatal):
        """Restore the excepted shape of the output by inserting PLACEHOLDER
        at the places where errors occurred. 

        Parameters
        ----------
        preds : dict
            Dictionary with predicted values from DelftaCalculator.predict
        len_input : int
            Length of the input list before removing faulty molecules
        fatal : [int]
            Indices of the faulty molecules

        Returns
        -------
        dict
            Dictionary with predicted values and placeholders for faulty molecules
        """
        idx_success = np.setdiff1d(np.arange(len_input), fatal)
        for key, val in preds.items():
            if key == "charges":
                idx_charge = 0
                temp = []
                for idx in range(len_input):
                    if idx in fatal:
                        temp.append(np.array(PLACEHOLDER))
                    else:
                        temp.append(val[idx_charge])
                        idx_charge += 1
            else:
                temp = np.zeros(len_input, dtype=np.float32)
                temp[idx_success] = val
                temp[fatal] = PLACEHOLDER
            preds[key] = temp
        return preds


if __name__ == "__main__":
    import argparse
    import json

    import pandas as pd
    from openbabel.pybel import Outputfile, readfile, readstring

    from delfta.utils import COLUMN_ORDER, preds_to_lists

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "smiles",
        type=str,
        nargs="?",
        default=None,
        help="SMILES string, readable with Openbabel",
    )
    parser.add_argument(
        "--infile",
        type=str,
        dest="infile",
        required=False,
        default=None,
        help="Path to a molecule file readable by Openbabel.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        dest="tasks",
        required=False,
        default=["all"],
        help="A list of tasks to predict. Available tasks include `[E_form, E_homo, E_lumo, E_gap, dipole, charges]`, by default 'all'.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        dest="outfile",
        required=True,
        help="Path to output filename.",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        default=False,
        action="store_true",
        help="Whether to write the results as csv file instead of the default json, by default False",
    )
    parser.add_argument(
        "--direct",
        dest="delta",
        default=True,
        action="store_false",
        help="Whether to use direct-learning models, by default `False`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        required=False,
        default=32,
        help="Batch size used for prediction of large files",
    )
    parser.add_argument(
        "--noforce3d",
        dest="force3d",
        default=True,
        action="store_false",
        help="Don't assign 3D coordinates to molecules lacking them",
    )
    parser.add_argument(
        "--noaddh",
        default=True,
        action="store_false",
        help="Don't add hydrogens to molecules lacking them",
        dest="addh",
    )
    parser.add_argument(
        "--xtbopt",
        dest="xtbopt",
        default=False,
        action="store_true",
        help="Whether to perform GFN2-xTB geometry optimization, by default False",
    )
    parser.add_argument(
        "--silent",
        dest="verbose",
        default=True,
        action="store_false",
        help="Disables stdout logger",
    )

    parser.add_argument(
        "--optmols_outfile",
        type=str,
        dest="optmols_outfile",
        required=False,
        default=False,
        help="Enables returning the optimized molecules (use in combination with xtbopt) to the specified file. Needs to be valid Openbabel output format.",
    )

    args = parser.parse_args()
    return_optmols = True if args.optmols_outfile else False

    if args.smiles is None and args.infile is None:
        raise ValueError(
            "Either a SMILES string or a path to an openbabel-readable file with the --infile flag should be provided."
        )


    if args.smiles is not None:
        input_ = readstring("smi", args.smiles)
    else:
        ext = os.path.splitext(args.infile)[1].lstrip(".")
        input_ = readfile(ext, args.infile)

    calc = DelftaCalculator(
        tasks=args.tasks,
        delta=args.delta,
        force3d=args.force3d,
        addh=args.addh,
        xtbopt=args.xtbopt,
        verbose=args.verbose,
        return_optmols=return_optmols,
    )

    if return_optmols:
        preds, opt_mols = calc.predict(input_, batch_size=args.batch_size)
    else:
        preds = calc.predict(input_, batch_size=args.batch_size)

    preds_list = preds_to_lists(preds)

    if args.csv:
        df = pd.DataFrame(preds)
        df = df[sorted(df.columns.tolist(), key=lambda x: COLUMN_ORDER[x])]
        df.to_csv(args.outfile, index=None)
    else:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(preds_list, f, ensure_ascii=False, indent=4)

    if args.optmols_outfile:
        ext = os.path.splitext(args.optmols_outfile)[1].lstrip(".")
        outfile = Outputfile(ext, args.optmols_outfile, overwrite=True)

        for mol in opt_mols:
            outfile.write(mol)
        outfile.close()
