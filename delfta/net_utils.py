from openbabel import pybel
import numpy as np
import torch

import networkx as nx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import add_self_loops


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

# TODO Ken: Add dict with final normalization values for multitask learning


class DeltaDataset(Dataset):
    def __init__(
        self, path_to_input_files="path_to_inputs.txt", input_format="xyz",
    ):

        with open(path_to_input_files, "r") as f:
            input_files = f.readlines()

        self.input_files = [x.strip("\n") for x in input_files]
        self.input_format = input_format

    def __getitem__(self, idx):

        # Load input file
        molecule = next(pybel.readfile(self.input_format, self.input_files[idx]))

        # Loop over atoms in molecule and get coordinates and atomic numbers
        coords = []
        atomids = []

        for atom in molecule:
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
        return len(self.input_files)


if __name__ == "__main__":

    train_data = DeltaDataset(
        path_to_input_files="/home/kenatz/CHEMBL1/conf_00/input_files.txt",
        input_format="xyz",
    )
    train_pbar = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=2)
    print("Training set: ", len(train_data))

    for g_batch in train_pbar:
        print(g_batch.atomids.size())
        print(g_batch.edge_index.size())
