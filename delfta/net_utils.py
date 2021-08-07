from collections import namedtuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected

hparam = namedtuple("hparam", ["n_outputs", "global_prop", "n_kernels", "mlp_dim"])

MULTITASK_ENDPOINTS = {"E_homo": 0, "E_lumo": 1, "E_gap": 2, "dipole": 3}

MODEL_HPARAMS = {
    "multitask_delta": hparam(4, True, 5, 256),
    "single_energy_delta": hparam(1, True, 4, 1024),
    "charges_delta": hparam(1, False, 5, 256),
    "wbo_delta": hparam(1, False, 5, 256),
    "multitask_direct": hparam(4, True, 5, 256),
    "single_energy_direct": hparam(1, True, 5, 256),
    "charges_direct": hparam(1, False, 5, 256),
    "wbo_direct": hparam(1, False, 5, 256),
}

QMUGS_ATOM_DICT = {
    1: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    15: 6,
    16: 7,
    17: 8,
    35: 9,
    53: 10,
}  # atomic number --> index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DelftaDataset(Dataset):
    def __init__(self, mols, wbo=False):
        """Base Dataset class for delfta

        Parameters
        ----------
        mols : [pybel.Molecule]
            A list of `pybel.Molecule` instances.
        """
        self.mols = mols
        self.wbo = wbo

    def __getitem__(self, idx):
        coords = []
        atomids = []

        mol = self.mols[idx]

        for atom in mol:
            coords.append(atom.coords)
            atomids.append(QMUGS_ATOM_DICT[atom.atomicnum])

        atomids = torch.LongTensor(np.array(atomids))
        coords = torch.FloatTensor(np.array(coords))

        # Get edges in from the fully connected graph
        if self.wbo:
            edge_index = []
            for bond in [mol.OBMol.GetBondById(i) for i in range(mol.OBMol.NumBonds())]:
                a1, a2 = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                edge_index.append([a1, a2])

            edge_index = (
                torch.from_numpy(np.array(edge_index) - 1).t().contiguous()
            )  ## obabel starts numbering atom idx at 1
        else:
            edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
            edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
            edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])

        # Graph object
        graph_data = Data(
            atomids=atomids,
            coords=coords,
            edge_index=edge_index,
            num_nodes=atomids.size(0),
            n_edges=edge_index.size(1),
        )

        return graph_data

    def __len__(self):
        return len(self.mols)

