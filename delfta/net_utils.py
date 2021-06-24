from collections import namedtuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected

hparam = namedtuple('hparam', ['n_outputs', 'global_prop'])

MULTITASK_ENDPOINTS = {"E_homo": 1, "E_lumo": 2, "E_gap": 3, "dipole": 4}

MODEL_HPARAMS = {
    "multitask_delta": hparam(5, True),
    "single_energy_delta": hparam(1, True),
    "charges_delta": hparam(1, False),
    "multitask_direct": hparam(5, True),
    "single_energy_direct": hparam(1, True),
    "charges_direct": hparam(1, False),
}

QMUGS_ATOM_DICT = {
    17: 8,
    35: 9,
    1: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    16: 7,
    15: 6,
    53: 10,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO Ken: Add dict with final normalization values for multitask learning


class DeltaDataset(Dataset):
    def __init__(self, mols):
        self.mols = mols

    def __getitem__(self, idx):
        coords = []
        atomids = []

        for atom in self.mols[idx]:
            coords.append(atom.coords)
            atomids.append(QMUGS_ATOM_DICT[atom.atomicnum])

        atomids = torch.LongTensor(np.array(atomids))
        coords = torch.FloatTensor(np.array(coords))

        # Get edges in from the fully connected graph
        edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
        edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
        edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])

        # Graph object
        graph_data = Data(
            atomids=atomids,
            coords=coords,
            edge_index=edge_index,
            num_nodes=atomids.size(0),
        )

        return graph_data

    def __len__(self):
        return len(self.mols)

