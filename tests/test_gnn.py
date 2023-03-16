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


# def test_gnn_model_invariance(device='cpu', dataset=MnistRotDataset, bs: int = 20,
#                               net_config: dict = {'k': 128, 'total_ds': 1., 'fill': 1., 'nbhd': 50,
#                                                   'group': SO2(0.)}, MAX_ERR: float = 0.3):
#     datasets = split_dataset(dataset(f'~/datasets/{dataset}/'), splits={'train': bs})
#
#     device = torch.device(device)
#
#     model = ImgGCNLieResnet(num_targets=datasets['train'].num_targets, **net_config).to(device)
#     model = torch.nn.Sequential(datasets['train'].default_aug_layers(), model)
#
#     dataloaders = {k: LoaderTo(DataLoader(v, batch_size=bs, shuffle=(k == 'train'),
#                                           num_workers=0, pin_memory=False), device) for k, v in datasets.items()}
#     data = next(iter(dataloaders['train']))[0]
#
#     with torch.inference_mode():
#         norm_res = model(data)
#         rot_data = data
#         for _ in range(3):
#             rot_data = rot_data.transpose(-2, -1).flip(-2)
#             rot_res = model(rot_data)
#
#             print(abs(norm_res - rot_res).max())
#             assert abs(norm_res - rot_res).max() < MAX_ERR, \
#                 'Error - too high error, model is not equivariant!'


def shift_x(coords):
    coords[1, :, 0] += 1  # shift x by 1

    return coords


def reflect_all(coords):
    coords[1] *= -1

    return coords


def get_rot_mat(angle):
    return torch.Tensor([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])


def rotate(coords, rot_ang: float = np.pi / 3):
    rot_mat = get_rot_mat(rot_ang)
    return torch.einsum('ij,klj->kli', rot_mat, coords)


def test_gnn_conv_equivariance(groups_and_trans: tuple = ((SO2(.05), rotate), (Trivial(), rotate)), bs: int = 3,
                               device='cpu', h: int = 10, w: int = 10, liftsamples: int = 1, n_mc_samples: int = 25,
                               ds_frac: float = 1., fill: float = 1., nbhd_size: int = 25, conv_layer=LieGNNSimpleConv):
    device = torch.device(device)

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    orig_coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float().view(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
    for group, trans in groups_and_trans:
        coords = orig_coords.clone()
        values = torch.randn((bs, coords.shape[1], 1))

        mask = torch.ones(bs, values.shape[1], device=device) > 0

        orig_graph = makeGraph((coords, values, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)
        coords = trans(coords)
        transf_graph = makeGraph((coords, values, mask), 1, group, nbhd_size=nbhd_size, liftsamples=liftsamples)

        # LieGNNSimpleConv only needs first two params
        gnn_conv = conv_layer(1, 1, mc_samples=n_mc_samples, ds_frac=ds_frac, bn=True, act='swish',
                              mean=True, group=group, fill=fill, cache=True, knn=False)

        orig_res = gnn_conv.forward(orig_graph.x,
                                    edge_index=orig_graph.edge_index,
                                    edge_attr=orig_graph.edge_attr)
        transf_res = gnn_conv.forward(transf_graph.x,
                                      edge_index=transf_graph.edge_index,
                                      edge_attr=transf_graph.edge_attr)

        assert torch.isclose(orig_res, transf_res).all(), f'Error - layer is not equivariant! Group - {group}'


if __name__ == "__main__":
    test_gnn_conv_equivariance()
