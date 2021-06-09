## Dataloader definitions and other util functions for neural network training/evaluation
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTITASK_ENDPOINTS = {"energy", "homo", "lumo", "gap", "dipole"}


# TODO: Combine single and multitask loader. To do Ken.
# TODO: Mol loader to do. Ken.
# TODO: Add pkl for max min values per property. {key: (min, max)...}

import networkx as nx
import numpy as np
import torch, h5py
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import add_self_loops


class DatasetSingletask(Dataset):
    def __init__(
        self,
        h5file="qt_dict.h5",
        prop="delta",
    ):
        self.h5f = h5py.File(h5file, "r")
        self.prop = prop

    def __getitem__(self, idx):

        # nodes coordinates and target
        atomids = torch.LongTensor(self.h5f[str(idx)]["atomids"])
        coords = torch.FloatTensor(self.h5f[str(idx)]["coords"])
        target = torch.FloatTensor(self.h5f[str(idx)][self.prop])

        # edges
        edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
        edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
        edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])

        # graph object
        graph_data = Data(
            atomids=atomids,
            coords=coords,
            edge_index=edge_index,
            target=target,
            num_nodes=atomids.size(0),
        )

        return graph_data

    def __len__(self):
        return len(self.h5f)


class DatasetMultitask(Dataset):
    def __init__(
        self,
        h5file="qt_dict.h5",
        prop="delta",
    ):
        self.h5f = h5py.File(h5file, "r")

        if "delta" in prop:
            print("Loading datset for delta learning")
            self.prop = [
                "DELTA_ENERGY",
                "DELTA_HOMO",
                "DELTA_LUMO",
                "DELTA_GAP",
                "DELTA_DIPOLE",
            ]
            self.prop_max = [
                1.780,
                0.1,
                0.313,
                0.250,
                0.935,
            ]
            self.prop_min = [
                0.289,
                0.05,
                0.269,
                0.176,
                -2.266,
            ]
        else:
            print("Loading datset for direct learning")
            self.prop = ["DFT_ENERGY", "DFT_HOMO", "DFT_LUMO", "DFT_GAP", "DFT_DIPOLE"]
            self.prop_max = [
                25.847,
                -0.248,
                0.0788,
                0.387,
                11.288,
            ]
            self.prop_min = [
                5.202,
                -0.335,
                -0.0473,
                0.238,
                0.798,
            ]

    def __getitem__(self, idx):
        
        # nodes and coordinates
        atomids = torch.LongTensor(self.h5f[str(idx)]["atomids"])
        coords = torch.FloatTensor(self.h5f[str(idx)]["coords"])

        # edges 
        edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
        edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
        edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])

        # normalized properties 
        energy = (
            torch.FloatTensor(self.h5f[str(idx)][self.prop[0]]) - self.prop_min[0]
        ) / (self.prop_max[0] - self.prop_min[0])
        energy = torch.FloatTensor(energy).unsqueeze(1)

        homo = (
            torch.FloatTensor(self.h5f[str(idx)][self.prop[1]]) - self.prop_min[1]
        ) / (self.prop_max[1] - self.prop_min[1])
        homo = torch.FloatTensor(homo).unsqueeze(1)

        lumo = (
            torch.FloatTensor(self.h5f[str(idx)][self.prop[2]]) - self.prop_min[2]
        ) / (self.prop_max[2] - self.prop_min[2])
        lumo = torch.FloatTensor(lumo).unsqueeze(1)

        gap = (
            torch.FloatTensor(self.h5f[str(idx)][self.prop[3]]) - self.prop_min[3]
        ) / (self.prop_max[3] - self.prop_min[3])
        gap = torch.FloatTensor(gap).unsqueeze(1)

        dipole = (
            torch.FloatTensor(self.h5f[str(idx)][self.prop[4]]) - self.prop_min[4]
        ) / (self.prop_max[4] - self.prop_min[4])
        dipole = torch.FloatTensor(dipole).unsqueeze(1)

        # concatenate properties to multitask taregt 
        target = torch.cat([energy, homo, lumo, gap, dipole], dim=1)

        # graph object
        graph_data = Data(
            atomids=atomids,
            coords=coords,
            edge_index=edge_index,
            target=target,
            num_nodes=atomids.size(0),
        )

        return graph_data

    def __len__(self):
        return len(self.h5f)
