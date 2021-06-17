import collections
import os
import pickle
import textwrap

import numpy as np
import torch
from torch_geometric.data.dataloader import DataLoader

from delfta.download import get_model_weights
from delfta.net import EGNN
from delfta.net_utils import MODEL_HPARAMS, MULTITASK_ENDPOINTS, DeltaDataset
from delfta.utils import LOGGER, MODEL_PATH
from delfta.xtb import run_xtb_calc
from delfta.net_utils import DEVICE


class DelftaCalculator:
    def __init__(self, tasks, delta=True, force3D=False) -> None:
        self.tasks = tasks
        self.delta = delta
        self.multitasks = [task for task in self.tasks if task in MULTITASK_ENDPOINTS]
        self.force3d = force3D

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

    def _preprocess(self, mols):
        idx_no3D = []
        for idx, mol in enumerate(mols):
            if mol.dim != 3:
                idx_no3D.append(idx)
                if self.force3d:
                    mol.make3D()
        if len(idx_no3D):
            if self.force3d:
                LOGGER.info(
                    f"Assigned MMFF94 coordinates to molecules with idx. {idx_no3D}"
                )

            else:
                raise ValueError(
                    textwrap.fill(
                        textwrap.dedent(
                            f"""
                Molecules at position {idx_no3D} have no 3D conformations available.
                Either provide a mol with one or re-run calculator with `force3D=True`.
                """
                        )
                    )
                )
        return mols

    def _get_preds(self, loader, model, scale=False):
        y_hats = []
        g_ptrs = []

        with torch.no_grad():
            for batch in loader:
                y_hats.append(model(batch).numpy())
                g_ptrs.append(batch.ptr.numpy())
        return y_hats, g_ptrs

    def _get_xtb_props(self, mols):
        xtb_props = collections.defaultdict(list)

        LOGGER.info("Now running xTB...")
        for mol in mols:
            xtb_out = run_xtb_calc(mol)
            for prop, val in xtb_out.items():
                xtb_props[prop].append(val)
        return xtb_props

    def _inv_scale(self, preds, norm_dict):
        return preds * norm_dict["scale"] + norm_dict["location"]

    def predict(self, mols, batch_size=32):
        mols = self._preprocess(mols)

        data = DeltaDataset(mols)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        preds = {}

        for _, model_name in enumerate(self.models):
            LOGGER.info(f"Now running network for model {model_name}...")
            model_param = MODEL_HPARAMS[model_name]
            model = EGNN(
                n_outputs=model_param.n_outputs, global_prop=model_param.global_prop
            ).eval()
            weights = get_model_weights(model_name).to(DEVICE)
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
            xtb_props = self._get_xtb_props(mols)

            for prop, delta_arr in preds_filtered.items():
                if prop == "charges":
                    preds_filtered[prop] = [
                        d_arr + np.array(xtb_arr)
                        for d_arr, xtb_arr in zip(delta_arr, xtb_props[prop])
                    ]
                preds_filtered[prop] = delta_arr + np.array(
                    xtb_props[prop], dtype=np.float32
                )
        return preds_filtered


if __name__ == "__main__":
    ## TODO: check if water raises a "not 3D" error
    from openbabel.pybel import readfile

    mols = [next(readfile("sdf", "data/trial/conf_final.sdf"))]

    calc = DelftaCalculator(
        tasks=["E_form", "E_homo", "E_lumo", "E_gap", "dipole"], delta=True
    )
    preds_delta = calc.predict(mols, batch_size=32)

    calc = DelftaCalculator(
        tasks=["E_form", "E_homo", "E_lumo", "E_gap", "dipole"], delta=False
    )
    preds_direct = calc.predict(mols, batch_size=32)

    xtb_props = run_xtb_calc(mols[0])

    ####
    from openbabel.pybel import readstring

    mols = [readstring("smi", "CCO")]
    calc = DelftaCalculator(
        tasks=["E_form", "E_homo", "E_lumo", "E_gap", "dipole"],
        delta=True,
        force3D=True,
    )
    preds_delta = calc.predict(mols, batch_size=32)

    # [mol.make3D() for mol in mols]

