import torch
import torch_geometric
from copy import deepcopy
import time

from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config
import copy
import lie_conv.lieGroups as lieGroups
import lie_conv.lieConv as lieConv
from lie_conv.lieConv import ImgLieGNN
from lie_conv.datasets import MnistRotDataset

def makeGraph(x, y, group, nbhd_size, liftsamples):
    lifted_x = group.lift(x, liftsamples)

    (_, vals, curr_mask) = x
    vals = vals[0]
    curr_mask = curr_mask[0]
    # (n, d) -> ( n, n, d)
    distances = group.distance(lifted_x[0])[0]

    # Returns list of nodes that are non-zero, i.e not masked
    nodes = torch.nonzero(curr_mask)[:, 0]
    # Get all possible combinations of nodes: [combs_cnt, 2]
    edge_pairs = torch.combinations(nodes)
    # Reflect each edge (as we need them in both directions)
    edge_pairs = torch.concat(
            [edge_pairs, deepcopy(edge_pairs)[:, [1, 0]]], dim=0) \
            .transpose(0, 1)
    # Extract only N nearest points
    if nbhd_size > 0:
        # Set up a mask to only selected edges to/from non-masked points
        mask = torch.ones(curr_mask.shape[0], curr_mask.shape[0], 
                dtype=torch.bool, device=vals.device)
        mask[edge_pairs[0], edge_pairs[1]] = 0
        # Set distance to masked points as inf
        distances[mask] = float('inf')
        # Find closest neighbours to each point
        ord_dist, ord_idx = torch.topk(distances, dim=-1, 
                largest=False, k=nbhd_size) 
        # Convert to the edge format
        # Extract all the indices
        rows_idx = torch.repeat_interleave(
                torch.arange(0, curr_mask.shape[0], device=vals.device), nbhd_size) 
        cols_idx = torch.arange(0, nbhd_size, 
                device=vals.device).repeat(curr_mask.shape[0])
        # Filter out indices that are masked
        cols_idx = cols_idx[torch.isin(rows_idx, nodes)]
        rows_idx = rows_idx[torch.isin(rows_idx, nodes)]
        
        # Have unidirected edges to n closest neighbours for each row
        edge_pairs = torch.stack([
            rows_idx,
            ord_idx[rows_idx, cols_idx].flatten()])

    # Use the pairs to extract distances
    edge_attr = distances[edge_pairs[0], 
            edge_pairs[1]][:, None]

    graph = torch_geometric.data.Data(
        x=vals, 
        edge_index=edge_pairs,
        edge_attr=edge_attr,
        y=y)

    return graph

def prepareImgToGraph(data, group, nbhd, liftsamples):
    x, y = data
    x = x[None, :]

    bs, c, h, w = x.shape

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()
    
    # Perform center crop
    # crop out corners (filled only with zeros)
    center_mask = coords.norm(dim=-1) < 15.
    coords = coords[center_mask] \
            .view(-1, 2).unsqueeze(0).repeat(bs, 1, 1).to(x.device)
    values = x.permute(0, 2, 3, 1)[:, center_mask, :] \
            .reshape(bs, -1, c)
    
    # all true
    mask = torch.ones(bs, values.shape[1], device=x.device) > 0  
    
    # new object to operate on:
    z = (coords, values, mask)

    return makeGraph(z, y, group, nbhd, liftsamples)  

def makeTrainer(*, dataset=MnistRotDataset, network=ImgLieGNN, 
                num_epochs=100, bs=50, lr=3e-3, 
                optim=Adam, device='cuda', trainer=Classifier,
                split={'train':12000}, small_test=False, 
                net_config={}, opt_config={},
                trainer_config={'log_dir':None}):

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'),
                             splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)
    graph_data = {}
    # Convert the datasets to graphs:
    for split, data in datasets.items():
        if split == 'test' and small_test:
            # have to manually limit size
            data = [data[idx] for idx in range(64)]
        graph_data[split]  = [prepareImgToGraph(data[idx], net_config['group'], 
                                  net_config['nbhd'], net_config['liftsamples']) 
                for idx in range(len(data))]

    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,
                    **net_config).to(device)
    model, bs = try_multigpu_parallelize(model,bs)
    dataloaders = {
            k:
                torch_geometric.loader.DataLoader(v,batch_size=bs, shuffle=(k=='train'),
                num_workers=0,pin_memory=False)
                for k,v in graph_data.items()}
    print(next(iter(dataloaders['train'])))
    dataloaders['Train'] = islice(dataloaders['train'],
                                  1+len(dataloaders['train'])//10)
    if small_test: 
        dataloaders['test'] = islice(dataloaders['test'],
                                     1+len(dataloaders['train'])//10)

    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)

    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__=="__main__":
    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save'] = False
    Trial(argupdated_config(defaults,namespace=(lieConv,lieGroups)))
