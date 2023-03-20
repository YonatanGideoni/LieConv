import copy
from functools import partial

import numpy as np
import torch
import torch_geometric

from lie_conv import lieConv, lieGroups, graphConv
from oil.tuning.args import argupdated_config

from oil.tuning.study import train_trial
from torch.optim import Adam
from torch_geometric.utils import to_undirected

from lie_conv.lieConv import ImgLieGNN
from oil.model_trainers import Regressor
from oil.utils.parallel import try_multigpu_parallelize
from oil.utils.utils import cosLr, islice, LoaderTo


def create_data(n_dims: int, n_points: int):
    return 2 * np.random.rand(n_dims, n_points) - 1


def data_to_graph(data: np.ndarray) -> torch_geometric.data.Data:
    norm_coords = torch.tensor(np.linalg.norm(data, axis=0))
    edge_pairs = torch.combinations(torch.Tensor(range(data.shape[1]))).swapaxes(0, 1).long()

    # TODO - try better distance metric than relative quadrant
    n_reflections = (-1) ** (data < 0).sum(axis=0)[:, None]
    pairwise_n_refl = n_reflections @ n_reflections.T
    edge_attr = torch.tensor(pairwise_n_refl)[edge_pairs[0], edge_pairs[1]]

    edge_pairs, edge_attr = to_undirected(edge_pairs, edge_attr.unsqueeze(-1), reduce='mean')
    return torch_geometric.data.Data(x=torch.tensor(data.swapaxes(0, 1)).float(), edge_index=edge_pairs,
                                     edge_attr=edge_attr, y=norm_coords.float())


def create_dataset(n_graphs: int, n_dims: int, n_points_per_graph: int):
    return [data_to_graph(create_data(n_dims, n_points_per_graph)) for _ in range(n_graphs)]


# TODO - make sure that the regressor properly handles node-level regression tasks
def makeTrainer(*, network=ImgLieGNN, num_epochs=100, bs=50, lr=3e-3, n_points_per_graph: int = 8, split: dict = {},
                optim=Adam, device='cuda', trainer=Regressor, small_test=False, net_config={}, opt_config={},
                trainer_config={'log_dir': None}):
    # Prep the datasets splits, model, and dataloaders
    n_dims = net_config['group'].q_dim
    print('Creating datasets')
    datasets = {'train': create_dataset(split['train'], n_dims, n_points_per_graph),
                'test': create_dataset(split['test'], n_dims, n_points_per_graph)}

    device = torch.device(device)
    model = network(chin=n_dims, num_targets=1, pool=False, **net_config).to(device)
    model, bs = try_multigpu_parallelize(model, bs)
    dataloaders = {
        k: LoaderTo(
            torch_geometric.loader.DataLoader(
                v, batch_size=bs, shuffle=(k == 'train'),
                num_workers=0, pin_memory=False),
            device) for k, v in datasets.items()
    }
    dataloaders['Train'] = islice(dataloaders['train'],
                                  1 + len(dataloaders['train']) // 10)
    if small_test:
        dataloaders['test'] = islice(dataloaders['test'],
                                     1 + len(dataloaders['train']) // 10)
    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)

    return trainer(model, dataloaders, opt_constr, lr_sched, **trainer_config)


if __name__ == '__main__':
    # n_dims = 2
    # n_points_per_graph = 5
    # n_graphs_per_dataset = 32
    # bs = 8
    #
    # train = create_dataset(n_graphs_per_dataset, n_dims, n_points_per_graph)
    # test = create_dataset(n_graphs_per_dataset, n_dims, n_points_per_graph)
    #
    # for _ in range(10):
    #     data = create_data(n_dims, n_points_per_graph)
    #
    #     graph = data_to_graph(data)
    #
    #     print()

    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save'] = False
    Trial(argupdated_config(defaults, namespace=(lieConv, lieGroups, graphConv)))
