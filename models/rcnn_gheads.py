import einops as ei
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances
from torchvision.models.detection import MaskRCNN
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.models.detection.roi_heads import *

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.ops as ops
from data.graph_builder import objgraph_to_objrelgraph

from models.rcnn_vrmn import _ObjPairLayer, _RoisPairExpandingLayer, RoiPairer
from models.rcnn_graph import RelCELoss
import torch_geometric as tg
from torch_geometric.nn import Sequential as Sequential_tg
from einops.layers.torch import Rearrange, Reduce

@ROI_HEADS_REGISTRY.register()
class GraphHead(StandardROIHeads):
    
    @staticmethod
    def gconv_block(in_feats, out_feats):
        return tg.nn.Sequential(
            'x, edge_index',
            [(tg.nn.GCNConv(in_feats, out_feats), 'x, edge_index -> x1'),
            torch.nn.ReLU(inplace=True),
            tg.nn.norm.BatchNorm(out_feats)
            ]
        )
    @staticmethod
    def gconv_cls(in_feats, out_feats):
        return tg.nn.GCNConv(in_feats, out_feats)
    
    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        depth=0
        self.pairer = _RoisPairExpandingLayer(separate=True)
        self.project_embed = Reduce('b c h w -> b c', 'mean')
        self.conv_obj = nn.ModuleList([self.gconv_block(256, 2048)] +
                                      [self.gconv_block(2048, 2048) for _ in range(depth)] +
                                      [self.gconv_cls(2048, 3)])
        self.conv_rel = nn.ModuleList([self.gconv_block(256, 2048)] +
                                      [self.gconv_block(2048, 2048) for _ in range(depth)] +
                                      [self.gconv_cls(2048, 3)])
        self.graph_loss = RelCELoss()

    @staticmethod
    def _build_graph(features: torch.Tensor, indices: List[List[Tuple[int]]], graph_gt: Optional[torch.Tensor]):
        """
        features: features in shape [N, C, W, H]
        indices: indeces of the graph in the form [BS, Nobj*(Nobj+1)/2, 2] 
        """
        features = ei.unpack(features, [[len(i)] for i in indices], '* C W H')
        batch = []
        for ix, (f, i) in enumerate(zip(features, indices)):
            i_tens = torch.tensor(i)
            n_rel, n_obj = len(i), torch.max(i_tens)
            arg = torch.arange(n_rel)
            a, b = torch.vstack([arg, i_tens[:, 0]])[:, n_obj+1:], torch.vstack([arg, i_tens[:, 1]])[:, n_obj+1:]
            edges = torch.concat([a, b], dim=1).to(device=f.device)
            is_object = torch.tensor([1]*(n_obj+1)+[0]*(n_rel- n_obj - 1), device=f.device, dtype=bool)
            if graph_gt is not None:
                g_gt = torch.zeros(f.shape[0], device=f.device).long()
                g_gt[f.shape[0]-graph_gt[ix].shape[0]:] = graph_gt[ix]
                batch.append(tg.data.Data(x=f, edge_index=edges, is_object=is_object, cls_gt=g_gt))
            else:
                batch.append(tg.data.Data(x=f, edge_index=edges, is_object=is_object))
        return tg.data.Batch.from_data_list(batch) # TODO Visualize and Check THIS
    
    def _forward_graph(self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor], gt_graph: Optional[List[tg.data.Data]]):
        bs = len(proposals)

        # Concatenate Proposals
        boxes, num_objs = ei.pack([p.tensor for p in proposals], "* b")
        device = boxes.device
        
        # Build Graphs
        if gt_graph is not None:
            index_to_node = [g.index_to_node for g in gt_graph]
            graphs = tg.data.Batch.from_data_list(gt_graph, exclude_keys=['index_to_node'])
        else:
            gen_graph_list = [objgraph_to_objrelgraph(num_objs=n[0]) for n in num_objs]
            index_to_node = [g.index_to_node for g in gen_graph_list]
            graphs = tg.data.Batch.from_data_list(gen_graph_list, exclude_keys=['index_to_node'])
        
        if graphs.num_nodes <= 1:
            graphs.x = torch.zeros(0, 3, device=device)
            return graphs
        
        # Extract Features for predictions
        features = [features[f] for f in self.box_in_features]
        device = features[0].device

            
        # Weird stuff
        batch_ix = torch.cat([torch.ones(j) * i for i, j in enumerate(num_objs)]).to(boxes.device)
        boxes, _ = ei.pack([batch_ix, boxes], "i *")
        paired_boxes, indices = self.pairer(boxes, bs, torch.tensor(num_objs), separate=True)
        paired_boxes = [Boxes(b[:, 1:])for b in paired_boxes]
        features = self.box_pooler(features, paired_boxes)
        tot = 0
        feat_index = []
        for i, j in zip(index_to_node, indices):
            feat_index.append(i+tot)
            tot += len(j)

        # for b in range(bs):
        #     [index_to_node[b][itn] for itn in indices[b]]
        graphs.x = features[torch.cat(feat_index)]
        graphs = graphs.to(device)
        is_object = graphs.is_object[:, None]
      
        graphs.x = self.project_embed(graphs.x)
        for conv_obj, conv_rel in zip(self.conv_obj, self.conv_rel):
            obj = conv_obj(graphs.x, graphs.edge_index)
            rel = conv_rel(graphs.x, graphs.edge_index)
            graphs.x = obj * is_object + rel * ~is_object
        
        is_object = is_object[:, 0]
        prediction = graphs.x
        if gt_graph is not None and self.training:
            return self.graph_loss(prediction[~is_object], graphs.rel_gt[~is_object].long())
        else:
            return graphs      
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        gt_graph: Optional[torch.Tensor] = None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            gt_boxes = [t.gt_boxes for t in targets]
        del targets
        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_graph(features, gt_boxes, gt_graph))
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # pred_graph = self._forward_graph(features, [x.pred_boxes for x in pred_instances], None)
            return pred_instances, {}
    
    @staticmethod  
    def pred_to_dense(pred: tg.data.Batch, num_objs: List[int]):
        res = []
        for i in range(len(pred)):
            ptr1, ptr2 = pred.ptr[i], pred.ptr[i+1]
            graph = pred[i]
            x = pred.x[ptr1:ptr2]
            num_objs = graph.is_object.sum()
            if num_objs <= 1:
                res.append(torch.tensor([]))
                continue
            dense = torch.zeros(num_objs, num_objs, x.shape[1], device=x.device)
            edge_index, _ = tg.utils.remove_self_loops(graph.edge_index)
            for i in range(num_objs):
                nodes, connectivity, _, _ = tg.utils.k_hop_subgraph(i, 2, edge_index, flow='source_to_target')
                connection_nodes = nodes[~graph.is_object[nodes]]
                src = connectivity[0, ::2]
                dest = connectivity[1, 1::2]
                dense[src, dest] = x[connection_nodes]
            dense = torch.argmax(dense, dim=2)
            dense = (dense == 1) | (dense.T == 2)
            res.append(dense)
        return res
        

class PositionalEncodingPaired(nn.Module):

    def __init__(self, d_model: int, max_len: int = 3):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, emb_dim, w, h]``
        """
        x = ei.rearrange(x, 'b l e -> l b e')
        x = x + self.pe[:x.size(0), :, :]
        x = ei.rearrange(x, 'l b e -> b l e')
        
        return x

from einops.layers.torch import Rearrange, Reduce


@ROI_HEADS_REGISTRY.register()
class GraphHeadGru(StandardROIHeads):

    def __init__(self, *args, in_dim=256, hidden_dim=256, out_dim=3, num_layers=5, **argv):
        super().__init__(*args, **argv)
        
        self.pairer = RoiPairer(self.box_pooler, duplicates=True)
        self.init_proj = nn.Sequential(
            Reduce('b n c w h -> b n c', 'mean'),
            PositionalEncodingPaired(256),
            Rearrange('b n c -> b (n c)'),
            nn.Linear(256 * 3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.gru_layer = tg.nn.GatedGraphConv(hidden_dim, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Linear(hidden_dim//2, 3)
        )
        self.graph_loss = RelCELoss()
    
    @staticmethod
    def _build_graph(i, device):
        n = torch.arange(i, device=device)
        return tg.data.Data(edge_index=torch.cartesian_prod(n, n).T, num_nodes=i)
        
    def _forward_graph(self, features: Dict[str, torch.Tensor],
                       proposals: List[torch.Tensor],
                       gt_graph: Optional[List[tg.data.Data]]):
        bs = len(proposals)
        features = [features[f] for f in self.box_in_features]
        paired_feats, indices = self.pairer(features, proposals)
        device = paired_feats.device
        if gt_graph is not None:
            graphs = tg.data.Batch.from_data_list(gt_graph)
        else:
            graphs = tg.data.Batch.from_data_list([self._build_graph(i.shape[0], device) for i in indices])
        graphs = graphs.to(device)
        
        if graphs.num_nodes == 0:
            graphs.y = torch.zeros(0, 3, device=device)
            return graphs
        
        x = self.init_proj(paired_feats) # Why embedding and then concatenation
        graphs.x = x
        
        x = self.gru_layer(graphs.x, graphs.edge_index)
        x = graphs.x
        y = self.cls_head(x)
        if gt_graph is not None and self.training:
            return self.graph_loss(y, graphs.rel_gt.long())
        else:
            graphs.x = y
            return graphs
    
    @staticmethod  
    def pred_to_dense(pred: tg.data.Batch, num_objs: List[int]):
        res = []
        curr = 0
        for n in num_objs:
            if n <= 1:
                res.append(torch.tensor([]))
                continue
            
            n_rel = (n*(n-1))//2
            y = torch.argmax(pred.x[curr:(curr+n_rel*2)], dim=1)
            y1, y2 = y[curr:(curr+n_rel)], y[curr+n_rel:(curr+n_rel*2)]
            i_x, i_y = np.triu_indices(n=n, k=1)
            dense1 = torch.zeros(n, n, dtype=y.dtype, device=y.device)
            dense2 = torch.zeros(n, n, dtype=y.dtype, device=y.device)
            dense1[i_x, i_y] = y1
            dense2[i_y, i_x] = y2
            dense = ((dense1 == 1) | (dense1.T == 2)) | ((dense2 == 1) | (dense2.T == 2)) 
            res.append(dense)
        return res
            

    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        gt_graph: Optional[torch.Tensor] = None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            gt_boxes = [t.gt_boxes for t in targets]
        del targets
        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_graph(features, gt_boxes, gt_graph))
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # pred_graph = self._forward_graph(features, [x.pred_boxes for x in pred_instances], None)
            return pred_instances, {}

        
        
        
    

