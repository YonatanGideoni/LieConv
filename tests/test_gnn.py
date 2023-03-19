import numpy as np
import torch
from torch.utils.data import DataLoader

from examples.train_gnn_img import makeGraph
from lie_conv.datasets import MnistRotDataset
from lie_conv.graphConv import LieGNNSimpleConv
from lie_conv.lieConv import ImgGCNLieResnet
from lie_conv.lieGroups import SO2, T, Trivial
from oil.datasetup.datasets import split_dataset
from oil.utils.utils import LoaderTo


def test_gnn_equiv():
    groups: tuple = (T(2), SO2(.05))
    bs: int = 1
    device = 'cpu'
    h: int = 1
    w: int = 1
    liftsamples: int = 1
    n_mc_samples: int = 25
    ds_frac: float = 1.
    fill: float = 1.
    nbhd_size: int = -1
    conv_layer = LieGNNSimpleConv
    device = torch.device(device)

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h + 1)
    j = torch.linspace(-w / 2., w / 2., w + 1)
    orig_coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float().view(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
    for group in groups:
        coords = orig_coords.clone() + 12

        vals1 = torch.tensor(np.array([[1], [1], [0], [0]])).unsqueeze(0)
        vals2 = torch.tensor(np.array([[0], [0], [1], [1]])).unsqueeze(0)
        mask = torch.ones(bs, vals1.shape[1], device=device) > 0
        graph1 = makeGraph((coords, vals1, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)
        graph2 = makeGraph((coords, vals2, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)

        # LieGNNSimpleConv only needs first two params
        gnn_conv = conv_layer(1, 1, mc_samples=n_mc_samples, ds_frac=ds_frac, bn=True, act='swish',
                              mean=True, group=group, fill=fill, cache=True, knn=False)

        res1 = gnn_conv.forward(graph1.x,
                                edge_index=graph1.edge_index,
                                edge_attr=graph1.edge_attr)
        res2 = gnn_conv.forward(graph2.x,
                                edge_index=graph2.edge_index,
                                edge_attr=graph2.edge_attr)

        if isinstance(group, T):
            assert np.isclose(res1[:2, 0].detach(), res2[2:, 0].detach()).all(), 'Error: T2 is not equivariant!'
        else:  # SO2
            assert not np.isclose(res1[:2, 0].detach(), res2[2:, 0].detach()).all(), \
                'Error - why the hell is SO2 equivariant here? Sometimes happens, super weird'
