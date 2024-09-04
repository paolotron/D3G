from typing import Optional
import torch_geometric as tg
import torch_geometric.data as data
import numpy as np
import torch

def objgraph_to_objrelgraph(*, obj_graph:Optional[data.Data] = None , num_objs:Optional[int]=None):
    
    if not ((obj_graph is None) ^ (num_objs is None)):
        raise TypeError("Only one argument can be specified")
    
    n_nodes = obj_graph.num_nodes if obj_graph is not None else num_objs
    n_rel = int(n_nodes * (n_nodes - 1) / 2)
    total_nodes = n_nodes + n_rel * 2
    is_object = torch.from_numpy(np.array([True] * n_nodes + [False] * n_rel * 2))
    
    rel_i, rel_j = np.triu_indices(n_nodes, 1)
    rel_i, rel_j = np.concatenate([rel_i, rel_j]), np.concatenate([rel_j, rel_i])
    
    roi_indices = np.triu_indices(n_nodes, 1)
    # indices = list(range(n_nodes))
    # index_to_feat = {frozenset(i): ix for ix, i in enumerate(zip(*roi_indices))}
    index_to_node = {i : ((i, i))for i in range(n_nodes)}
    index_to_node = {**{j: ((rel_i[i], rel_j[i])) for i, j in enumerate(range(n_nodes, total_nodes))}, **index_to_node}
    node_to_index = {v: k for k, v in index_to_node.items()}
    edge_to_feat = {**{frozenset(i): ix+n_nodes for ix, i in enumerate(zip(*roi_indices))},
                    **{frozenset([i]): i for i in range(n_nodes)}}
    feat_indices = torch.tensor([edge_to_feat[frozenset(index_to_node[i])] for i in range(len(index_to_node))])
    edges = []
    for (start, dest), index in node_to_index.items():
        if start == dest:
            edges.append((index, index))
        else:
            edges.append((start, index))
            edges.append((index, dest))
    edges = torch.Tensor(edges)
    rel_graph = data.Data(num_nodes=total_nodes, edge_index=edges.T.long(), is_object=is_object, index_to_node=feat_indices)
    if obj_graph is not None:
        rel_gt = torch.zeros(rel_graph.num_nodes)
        if obj_graph.edge_index is not None:
            for edge in obj_graph.edge_index.T:
                edge = tuple(edge.long().tolist())
                rel_gt[node_to_index[edge]] = 1
                rel_gt[node_to_index[tuple(reversed(edge))]] = 2
        rel_graph.rel_gt = rel_gt
    return rel_graph
    
    