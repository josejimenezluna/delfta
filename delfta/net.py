"""
© 2021, ETH Zurich
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor
from torch_scatter import scatter_mean


class EGNN(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=32,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
        global_prop=True,
        scatter_fun=scatter_mean
    ):
        """Main Equivariant Graph Neural Network class.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        n_outputs : int, optional
            Number of endpoints to predict, by default 1
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Aggregation strategy for global tasks, by default "mean"
        global_prop : bool, optional
            Whether to predict a molecule-level property or an atomic one, by default True
        scatter_fun : function, optional
            Which torch.scatter function to use in order to aggregate node-level features
        """
        super(EGNN, self).__init__()

        self.pos_dim = 3
        self.m_dim = m_dim
        self.embedding_dim = embedding_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr
        self.global_prop = global_prop
        self.scatter_fun = scatter_fun

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=11, embedding_dim=self.embedding_dim
        )

        # Kernel
        self.kernel_dim = self.embedding_dim
        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP 1
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))

        if self.global_prop:
            # MLP 2
            self.fnn2 = nn.ModuleList()
            for _ in range(self.n_mlp - 1):
                self.fnn2.append(nn.Linear(self.mlp_dim, self.mlp_dim))
            self.fnn2.append(nn.Linear(self.mlp_dim, self.n_outputs))

        else:
            self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            if self.global_prop:
                self.fnn2.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, g_batch):
        # Embedding
        features = self.embedding(g_batch.atomids)
        features = torch.cat([g_batch.coords, features], dim=1)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(x=features, edge_index=g_batch.edge_index,)

            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(torch.cat(feature_list, dim=1))

        # MLP 1
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))

        features = self.fnn[-1](features)
        if self.global_prop:
            # Pooling
            features = F.silu(features)
            features = self.scatter_fun(features, g_batch.batch, dim=0)

            # MLP 2
            for mlp in self.fnn2[:-1]:
                features = F.silu(mlp(features))

            # Outputlayer
            features = self.fnn2[-1](features)
        else:
            features = features.squeeze(1)
        return features


def weights_init(m):
    """Xavier uniform weight initialization

    Parameters
    ----------
    m : [torch.nn.modules.linear.Linear]
        A list of learnable linear PyTorch modules.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class EGNN_sparse(MessagePassing):
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=32,
        dropout=0.1,
        fourier_features=32,
        aggr="mean",
        **kwargs,
    ):
        """Base torch geometric EGNN message-passing layer

        Parameters
        ----------
        feats_dim : int
            Dimension of the node-level features
        pos_dim : int, optional
            Dimensions of the positional features (e.g. cartesian coordinates), by default 3
        edge_attr_dim : int, optional
            Dimension of the edge-level features, by default 0
        m_dim : int, optional
            Hidden node/edge layer size, by default 32
        dropout : float, optional
            Whether to use dropout, by default 0.1
        fourier_features : int, optional
            Number of Fourier features, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        """
        valid_aggrs = {
            "add",
            "sum",
            "max",
            "mean",
        }
        assert aggr in valid_aggrs, f"pool method must be one of {valid_aggrs}"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        # Model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features

        self.edge_input_dim = (
            (self.fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge layers
        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        # Node layers
        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        # Initialization
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, x: Tensor, edge_index: Adj,
    ):

        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(
                rel_dist, num_encodings=self.fourier_features
            )
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        hidden_out = self.propagate(
            edge_index, x=feats, edge_attr=rel_dist, coors=coors, rel_coors=rel_coors,
        )

        return torch.cat([coors, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        update_kwargs = self.inspector.collect_param_data("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm1(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out, **update_kwargs)


class EGNNWBO(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        n_kernels=5,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=64,
        initialize_weights=True,
        fourier_features=32,
        aggr="mean",
    ):
        """WBO equivariant Graph Neural Network class. This is to be deprecated
        on future releases.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding dimension, by default 128
        n_kernels : int, optional
            Number of message-passing rounds, by default 5
        n_mlp : int, optional
            Number of node-level and global-level MLPs, by default 3
        mlp_dim : int, optional
            Hidden size of the node-level and global-level MLPs, by default 256
        n_outputs : int, optional
            Number of endpoints to predict, by default 1
        m_dim : int, optional
            Node-level hidden size, by default 32
        initialize_weights : bool, optional
            Whether to use Xavier init. for learnable weights, by default True
        fourier_features : int, optional
            Number of Fourier features to use, by default 32
        aggr : str, optional
            Aggregation strategy for global tasks, by default "mean"
        global_prop : bool, optional
            Whether to predict a molecule-level property or an atomic one, by default True
        """
        super(EGNNWBO, self).__init__()

        self.pos_dim = 3
        self.m_dim = m_dim
        self.embedding_dim = embedding_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=11, embedding_dim=self.embedding_dim
        )

        # Node kernel
        self.kernel_dim = self.embedding_dim
        self.kernels = nn.ModuleList()
        self.edge_kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse_edge(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP 1
        self.fnn = nn.ModuleList()
        input_fnn = self.m_dim * self.n_kernels
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, g_batch):
        # Embedding
        features = self.embedding(g_batch.atomids)
        features = torch.cat([g_batch.coords, features], dim=1)

        # Kernel
        feature_list = []

        for i in range(len(self.kernels)):
            features, m_ij = self.kernels[i](x=features, edge_index=g_batch.edge_index,)
            feature_list.append(m_ij)

        # concat
        features = F.silu(torch.cat(feature_list, dim=1))

        # mlp 1
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))

        # outputlayer
        features = self.fnn[-1](features).squeeze(1)

        return features


class EGNN_sparse_edge(MessagePassing):
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=32,
        dropout=0.1,
        fourier_features=32,
        aggr="mean",
        **kwargs,
    ):
        """Base torch geometric WBO EGNN message-passing layer

        Parameters
        ----------
        feats_dim : int
            Dimension of the node-level features
        pos_dim : int, optional
            Dimensions of the positional features (e.g. cartesian coordinates), by default 3
        edge_attr_dim : int, optional
            Dimension of the edge-level features, by default 0
        m_dim : int, optional
            Hidden node/edge layer size, by default 32
        dropout : float, optional
            Whether to use dropout, by default 0.1
        fourier_features : int, optional
            Number of Fourier features, by default 32
        aggr : str, optional
            Node update function, by default "mean"
        """
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse_edge, self).__init__(**kwargs)

        # Model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features

        self.edge_input_dim = (
            (self.fourier_features * 2) + edge_attr_dim + 1 + feats_dim
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge layers
        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        # Node layers
        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        # Initialization
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, x: Tensor, edge_index: Adj,
    ):

        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        # add second dist

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(
                rel_dist, num_encodings=self.fourier_features
            )
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        hidden_out, m_ij = self.propagate(
            edge_index, x=feats, edge_attr=rel_dist, coors=coors, rel_coors=rel_coors,
        )

        return torch.cat([coors, hidden_out], dim=-1), m_ij

    def message(self, x_i, x_j, edge_attr):
        x_ij = torch.sum(torch.cat([x_i.unsqueeze(2), x_j.unsqueeze(2)], dim=2), dim=2)
        m_ij = self.edge_mlp(torch.cat([x_ij, edge_attr], dim=1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):

        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)
        m_ij_mirrored = torch.cat([m_ij, m_ij], dim=0)

        # get new edge ids
        edge_mirrored = torch.cat(
            (edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)), dim=0
        )
        edge_index = torch.cat([edge_index, edge_mirrored], dim=1)

        # update dict with new edge ids
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        update_kwargs = self.inspector.collect_param_data("update", coll_dict)

        # aggregate messages
        m_i = self.aggregate(m_ij_mirrored, **aggr_kwargs)
        m_i = self.edge_norm1(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out, **update_kwargs), m_ij
