import numpy as np
import torch
import torch_geometric.data

from examples.train_gnn_img import makeGraph
from lie_conv.lieConv import LieGNN
from lie_conv.lieGroups import Tx


def test_gnn_orbit_pooling():
    group = Tx(2)
    bs: int = 1
    device = 'cpu'
    h: int = 1
    w: int = 1
    liftsamples: int = 1
    nbhd_size: int = -1
    device = torch.device(device)

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h + 1)
    j = torch.linspace(-w / 2., w / 2., w + 1)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float().view(-1, 2).unsqueeze(0).repeat(bs, 1, 1) + 12

    vals1 = torch.tensor(np.array([[1], [1], [0], [0]])).unsqueeze(0).float()
    vals2 = torch.tensor(np.array([[0], [0], [1], [1]])).unsqueeze(0).float()
    mask = torch.ones(bs, vals1.shape[1], device=device) > 0
    graph1 = makeGraph((coords, vals1, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)
    graph2 = makeGraph((coords, vals2, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)

    gnn = LieGNN(chin=1, group=group, num_layers=0, agg_orbits=True, num_orbits=2)
    gnn.eval()

    res1 = gnn.forward(torch_geometric.data.Batch.from_data_list([graph1]))
    res2 = gnn.forward(torch_geometric.data.Batch.from_data_list([graph2]))

    assert (res1 == res2).all()
