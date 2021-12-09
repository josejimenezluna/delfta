"""
© 2021, ETH Zurich
"""

from collections import namedtuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected
from torch_scatter import scatter_mean, scatter_sum


hparam = namedtuple("hparam", ["n_outputs", "global_prop", "n_kernels", "mlp_dim", "scatter_fun"])

MULTITASK_ENDPOINTS = {"E_homo": 0, "E_lumo": 1, "E_gap": 2, "dipole": 3}

MODEL_HPARAMS = {
    "multitask_delta": hparam(4, True, 5, 256, scatter_mean),
    "single_energy_delta": hparam(1, True, 5, 256, scatter_sum),
    "charges_delta": hparam(1, False, 5, 256, scatter_mean),
    "wbo_delta": hparam(1, False, 5, 256, scatter_mean),
    "multitask_direct": hparam(4, True, 5, 256, scatter_mean),
    "single_energy_direct": hparam(1, True, 5, 256, scatter_sum),
    "charges_direct": hparam(1, False, 5, 256, scatter_mean),
    "wbo_direct": hparam(1, False, 5, 256, scatter_mean),
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
    def __init__(self, mols):
        """Base Dataset class for delfta

        Parameters
        ----------
        mols : [pybel.Molecule]
            A list of `pybel.Molecule` instances.
        """
        self.mols = mols

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
        edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
        edge_index = torch.from_numpy(edge_index).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])
        edge_index = to_undirected(edge_index)

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

