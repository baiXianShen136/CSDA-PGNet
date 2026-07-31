from random import random

import torch
from torch_geometric.utils import degree, to_undirected

import networkx as nx
from torch_scatter import scatter
from torch_geometric.utils import to_networkx


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def drop_edge_weighted_low(edge_index, edge_weights, p: float, threshold: float = 1.):

    edge_weights = normalize_edge_weights(edge_weights)

    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def normalize_edge_weights(edge_weights):
    min_weight = edge_weights.min()
    max_weight = edge_weights.max()

    normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)

    return normalized_weights

def drop_edge_weighted_new(edge_index, edge_weights, pr_drop_weights, edgenet_input , p: float, threshold: float = 1.):
    # min_weight = edge_weights.min()
    # max_weight = edge_weights.max()


    normalized_weights = normalize_edge_weights(edge_weights)

    edge_weights = p * normalized_weights + (1 - p) * pr_drop_weights

    # print(f"edge_weights: {edge_weights}")

    if torch.isnan(edge_weights).any():
        print("NaN detected after combining weights")
        print(f"edge_weights: {edge_weights}")
        print(f"normalized_weights: {normalized_weights}")
        print(f"pr_drop_weights: {pr_drop_weights}")


    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)


    return edge_index[:, sel_mask],edgenet_input[sel_mask]

def pr_drop_weights_new(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr_new(edge_index, k=k)

    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)


    s_row = torch.zeros_like(pv_row)
    s_col = torch.zeros_like(pv_col)

    non_zero_mask_row = pv_row > 0
    non_zero_mask_col = pv_col > 0
    s_row[non_zero_mask_row] = torch.log(pv_row[non_zero_mask_row])
    s_col[non_zero_mask_col] = torch.log(pv_col[non_zero_mask_col])


    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

   

    weights = (s - s.min()) / (s.max() - s.min())


    return weights


def compute_pr_new(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    

    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    # print(f"befer x: {x}")
    for i in range(k):
       
        safe_deg_out = deg_out.clone()
        safe_deg_out[safe_deg_out == 0] = 1
        # print(f"safe_deg_out: {safe_deg_out}")

     
        edge_msg = x[edge_index[0]] / safe_deg_out[edge_index[0]]
        edge_msg[deg_out[edge_index[0]] == 0] = 0
        # print(f"edge_msg: {edge_msg}")

        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        # print(f"agg_msg: {agg_msg}")

        x = (1 - damp) * x + damp * agg_msg
        # print(f"Iteration {i} - x: {x}")

    return x

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph, tol=1e-8)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)



def random_select_and_zero_out(self, node_ftr, p, dim_to_zero, edge_index):
    num_nodes, num_dims = node_ftr.shape
    num_selected = int(p * num_nodes)
    selected_indices = random.sample(range(num_nodes), num_selected)
    new_node_ftr = node_ftr.clone()
    for index in selected_indices:
        new_node_ftr[index, dim_to_zero] = 0
    return new_node_ftr, edge_index
