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


def trans_x_vals(coords):
    trans_coords = coords.clone()
    trans_coords[:, :, 0] += 1
    return (coords ** 2).sum(axis=-1).unsqueeze(-1), (trans_coords ** 2).sum(axis=-1).unsqueeze(-1), trans_coords


def test_gnn_conv_equivariance(groups_and_coords2vals: tuple = ((T(2), trans_x_vals), (Trivial(), trans_x_vals)),
                               bs: int = 1, device='cpu', h: int = 10, w: int = 10, liftsamples: int = 1,
                               n_mc_samples: int = 25, ds_frac: float = 1., fill: float = 1., nbhd_size: int = -1,
                               conv_layer=LieGNNSimpleConv):
    device = torch.device(device)

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h + 1)
    j = torch.linspace(-w / 2., w / 2., w + 1)
    orig_coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float().view(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
    for group, vals_from_coords in groups_and_coords2vals:
        coords = orig_coords.clone()
        values = torch.randn((bs, coords.shape[1], 1))

        mask = torch.ones(bs, values.shape[1], device=device) > 0

        vals1, vals2, trans_coords = vals_from_coords(coords)
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

        coord_val_map1 = {tuple(coord): float(val) for coord, val in
                          zip(coords.flatten(end_dim=-2).numpy(), res1.flatten())}
        coord_val_map2 = {tuple(coord): float(val) for coord, val in
                          zip(trans_coords.flatten(end_dim=-2).numpy(), res2.flatten())}
        print(abs(np.array([coord_val_map1[c1] - coord_val_map2[c2]
                            for c1, c2 in zip(coord_val_map1, coord_val_map2)])).mean(), group)
        print()
        # assert torch.isclose(res1, res2).all(), f'Error - layer is not equivariant! Group - {group}'


if __name__ == "__main__":
    test_gnn_conv_equivariance()
