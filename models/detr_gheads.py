import numpy as np
import torch
import torch.nn as nn

from models.detr_modules.detr import MLP
from models.detr_modules.transformer import TransformerDecoder, TransformerDecoderLayer
from models.graph_transformer_dense import GraphTransformerLayerDense



import networkx as nx
import einops as es
import einops.layers.torch as el
from detectron2.config import configurable

class PairwiseConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        d1, d2 = x.shape[-2], y.shape[-2]
        grid_x, grid_y = torch.meshgrid(torch.arange(d1, device=x.device), torch.arange(d2, device=y.device), indexing='ij')
        res = torch.concat([torch.index_select(x, dim=-2, index=grid_x.flatten()), torch.index_select(y, dim=-2, index=grid_y.flatten())], dim=-1)
        res = es.rearrange(res, '... (L1 L2) C -> ... L1 L2 C', L1=d1, L2=d2)
        return res
        

class DummyHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, hs):
        hs = hs ** 2

class BaseGHead(nn.Module):
    
    @configurable
    def __init__(self, num_layers=1, in_dim=256, hidden_dim=256, num_heads=1, num_nodes=100, edge_features='constant_one') -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.edge_features = edge_features
        
        if edge_features == 'concat':
            out_proj_edge = hidden_dim // 2
            self.pairwise_layer = PairwiseConcatLayer()
        else:
            out_proj_edge = hidden_dim 

        
        self.proj_e1 = nn.Linear(in_dim, out_proj_edge)
        self.proj_e2 = nn.Linear(in_dim, out_proj_edge)
        
        self.proj_node_input = nn.Linear(in_dim, hidden_dim)
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
    
    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.MODEL.GRAPH_HEAD
        return {'num_layers': cfg.NUM_LAYERS, 'hidden_dim': cfg.HIDDEN_DIM, 'num_heads': cfg.NUM_HEADS, 'edge_features': cfg.EDGE_FEATURES} 
    
    
    def _compute_edge_features(self, features):
        # features L B Q C 
        L, B, Q, C = features.shape
        device = features.device
        C = self.hidden_dim
        e1, e2 = self.proj_e1(features), self.proj_e2(features)
        if self.edge_features == 'concat':
            e = self.pairwise_layer(e1, e2)
        elif self.edge_features == 'sum':
            e = e1[:, :, None, :, :] + e2[:, :, :, None, :]
        elif self.edge_features == 'diff': 
            e = e1[:, :, None, :, :] - e2[:, :, :, None, :]
        elif self.edge_features == 'div':
            e = e1[:, :, None, :, :] / e2[:, :, :, None, :]
        elif self.edge_features == 'mul':
            e = e1[:, :, None, :, :] * e2[:, :, :, None, :]
        else:
            raise NotImplementedError(f'{self.edge_features} aggregations not implemented')
        return e
    

class DenseGraphTransformerHead(BaseGHead):
    
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_transformer_layers = nn.ModuleList([
            GraphTransformerLayerDense(in_dim=self.hidden_dim, out_dim=self.hidden_dim, num_heads=self.num_heads, layer_norm=True, batch_norm=False)
        for _ in range(self.num_layers)
        ])
        self.edge_cls = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim//2, output_dim=1, num_layers=3)
    
    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.MODEL.GRAPH_HEAD
        return {'num_layers': cfg.NUM_LAYERS, 'hidden_dim': cfg.HIDDEN_DIM, 'num_heads': cfg.NUM_HEADS, 'edge_features': cfg.EDGE_FEATURES} 
    
    
    def forward(self, hs: torch.Tensor):
        """ _summary_
        Args:
            hs (torch.Tensor): L x B x Q x C
        """
        L, B, Q, C = hs.shape 
        
        e = self._compute_edge_features(features=hs)
        hs = self.proj_node_input(hs)
        # hs = es.rearrange(hs, 'B L Q C -> (B L) Q C')
        
        
        # e = es.rearrange(e, '(B Q1 Q2) C -> B Q1 Q2 C', B=L*B, Q1=Q, Q2=Q, C=C)
        hs = es.rearrange(hs, 'L B Q C -> (L B) Q C')
        e = es.rearrange(e, 'L B Q1 Q2 C -> (L B) Q1 Q2 C')
        
        for layer in self.graph_transformer_layers:
            hs, e = layer(hs, e)
        e = self.edge_cls(e)
        e = es.rearrange(e, "(L B) Q1 Q2 C -> L B Q1 Q2 C", C=1, L=L, Q1=Q, Q2=Q, B=B)
        return e


def build_graph_head(cfg):
    name = cfg.MODEL.GRAPH_HEAD.NAME
    head = {
        'DummyHead': DummyHead,
        'GraphTransformerDense': DenseGraphTransformerHead,
    }[name](cfg)
    return head
    
    