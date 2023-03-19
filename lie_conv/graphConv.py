import torch_geometric
import torch

import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from lie_conv.utils import Expression, export


@export
class LieGNNSimpleConv(MessagePassing):
    """
        Perform simple equivariant convolution:

        h_u = \phi (\sum_v d((u_i, q_i), (v_j, q_j)) * h^{l-1}_v)

        where d^2 = ||log(v^{-1}u)||^2 + \alpha ||q_i - q_j||^2
        (the default distance in the LieConv paper)
    """

    def __init__(self, c_in, c_out, hidden_dim=None, agg='add', enable_mlp: bool = True, **kwargs):
        super().__init__(aggr=agg)
        if hidden_dim is None:
            self.hidden_dim = c_out
        else:
            self.hidden_dim = hidden_dim

        if enable_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(c_in, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, c_out),
                nn.ReLU()
            )
        else:
            self.mlp = lambda x: x

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        """
        x_j: (e, d_h) values at the source node i
        edge_attr: (e, d_e) edge attributes
        Calculate product of distance and hidden representations
        """
        messages = torch.einsum("ij,ik->ij", x_j, edge_attr)
        return messages

    def aggregate(self, inputs, index, dim_size):
        """
        Aggregate messages from all neighbouring nodes
        
        inputs: (e, d_h) messages m_ij for each node
        index: (e, 1) destination nodes for each message
        """
        aggr_out = scatter(inputs,
                           index,
                           dim=self.node_dim,
                           reduce=self.aggr,
                           dim_size=dim_size)  # dim_size to ensure even nodes are padded,, we return same size
        return aggr_out

    def update(self, aggr_out):
        """
        Apply MLP to the convolved values
        aggr_out: (n, d_h) convolved values
        """
        return self.mlp(aggr_out)


@export
class LieConvGCN(LieGNNSimpleConv):
    def __init__(self, c_in, c_out, hidden_dim=None, agg='add', edge_dim=1, **kwargs):
        super().__init__(c_in, c_out, hidden_dim, agg, **kwargs)

        self.mlp_msg = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.ReLU()
        )

    def message(self, x_i, x_j, edge_attr):
        embedded_edge_attr = self.mlp_msg(edge_attr)
        messages = torch.einsum("ij,ik->ij", x_i, edge_attr)
        return messages
