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

from models.rcnn_graph import RelCELoss
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.ops as ops

class RoiPairer(nn.Module):
    def __init__(self, box_pooler: nn.Module, duplicates=False) -> None:
        super().__init__()
        self.roi_expander = _RoisPairExpandingLayer()
        self.roi_align = ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.roi_pair = _ObjPairLayer()
        self.box_pooler = box_pooler
        self.duplicates = duplicates
    
    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        bs = len(proposals)
        boxes, num_objs = ei.pack([p.tensor for p in proposals], "* b")
        device = boxes.device
        
        if sum(map(lambda x: x[0], num_objs)) <= 1: # No Preds
            return torch.zeros(0, 3, device=device), [torch.zeros(0, device=device) for _ in range(bs)]
        
        batch_ix = torch.cat([torch.ones(j) * i for i, j in enumerate(num_objs)]).to(boxes.device)
        boxes, _ = ei.pack([batch_ix, boxes], "i *")
        expanded_rois, indices = self.roi_expander(boxes, bs, torch.tensor(num_objs), separate=True)
        # First N Rois are for objects then the others are the combinations of size 2
        expanded_rois = torch.concatenate(expanded_rois, dim=0)
        batch_shapes = [i.shape[0:1] for i in indices]
        expanded_rois = ei.unpack(expanded_rois[:, 1:], batch_shapes, '* i')
        expanded_rois = [Boxes(b) for b in expanded_rois]
        relationship_feats = self.box_pooler(features, expanded_rois)
        if not self.duplicates:
            relationship_feats = self.roi_pair(relationship_feats, bs, torch.tensor(num_objs))
            relation_indices = [np.vstack(np.triu_indices(i[0], 1)) for i in num_objs]
        else:
            start = 0
            tot_indices = []
            tot_indices_independent = []
            for b, n in zip(indices, num_objs):
                n = n[0]
                pair_ixes = torch.arange(n, b.shape[0]).reshape(-1, 1)
                i1 = torch.concatenate([b[n:], pair_ixes], dim=1)
                i2 = torch.concatenate([b[n:, [1, 0]], pair_ixes], dim=1)
                iall = torch.vstack([i1, i2])
                tot_indices.append(iall + start)
                tot_indices_independent.append(iall)
                start += b.shape[0]
                
            tot_indices = torch.vstack(tot_indices)
            relationship_feats = relationship_feats[tot_indices]
            relation_indices = tot_indices_independent
                
        return relationship_feats, relation_indices
        


@ROI_HEADS_REGISTRY.register()
class VMN_Head(StandardROIHeads):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.roi_expander = _RoisPairExpandingLayer()
        self.roi_align = ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.roi_pair = _ObjPairLayer()

        self.relation_branch = nn.Sequential(
            nn.Conv2d(256 * 3, 256*3, 1, groups=3),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(),
            nn.Conv2d(256 * 3, 256 * 3, 3, groups=3),
            nn.BatchNorm2d(256 * 3),
            nn.ReLU(), 
            nn.Conv2d(256 * 3, 512 * 3, 1, groups=3),
            nn.BatchNorm2d(512 * 3),
            nn.ReLU(),
            nn.Conv2d(512 * 3, 64 * 3, 3, groups=3),
            nn.BatchNorm2d(64 * 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.relation_head = nn.Sequential(
            nn.Linear(64 * 3, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 3)
        )
        self.loss = RelCELoss()

    def _forward_graph(self, features, proposals, gt_graph):
        bs = len(proposals)
        
        features = [features[f] for f in self.box_in_features]
        boxes, num_objs = ei.pack([p.tensor for p in proposals], "* b")
        device = features[0].device
        
        if sum(map(lambda x: x[0], num_objs)) <= 1: # No Preds
            return [torch.zeros(0, 3, device=device) for _ in range(bs)]
        
        batch_ix = torch.cat([torch.ones(j) * i for i, j in enumerate(num_objs)]).to(boxes.device)
        boxes, _ = ei.pack([batch_ix, boxes], "i *")
        expanded_rois, _ = self.roi_expander(boxes, bs, torch.tensor(num_objs)) # First N Rois are for objects then the others are the combinations of size 2
        batch_shapes = [torch.sum(expanded_rois[:, 0] == i).unsqueeze(0) for i in range(bs)]
        expanded_rois = ei.unpack(expanded_rois[:, 1:], batch_shapes, '* i')
        expanded_rois = [Boxes(b) for b in expanded_rois]
        # plot_sample(x[0], pred_detection=gt[0]['boxes'])

        relationship_feats = self.box_pooler(features, expanded_rois)
        relationship_feats = self.roi_pair(relationship_feats, bs, torch.tensor(num_objs))
        relationship_feats = self.relation_branch(ei.rearrange(relationship_feats, 'b n c w h -> b (n c) w h'))
        relationship_feats = self.relation_head(relationship_feats[..., 0, 0])
        if self.training:
            gt_graph = torch.cat(gt_graph).long().to(relationship_feats.device)
            return self.loss(relationship_feats, gt_graph)
        else:
            # upper triangular
            splits = [int((i[0]**2-i[0])/2) for i in num_objs]
            splitted_feats = torch.split(relationship_feats, splits)
            return splitted_feats

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        gt_graph: Optional[torch.Tensor] = None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
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
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            if not all([len(i) == 0 for i in  gt_boxes]):
                losses.update(self._forward_graph(features, gt_boxes, gt_graph))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # pred_graph = self._forward_graph(features, [x.pred_boxes for x in pred_instances], None)
            return pred_instances, {}
    
    @staticmethod
    def pred_to_dense(pred: List[torch.Tensor], num_objs: List[int]):
        res = []
        running_ix = 0
        for p, num in zip(pred, num_objs):
            if num <= 1:
                res.append(torch.tensor([]))
                continue
            i_x, i_y = np.triu_indices(n=num, k=1)
            curr_pred = p
            curr_pred = torch.argmax(curr_pred, axis=1)
            dense = torch.zeros(num, num, dtype=curr_pred.dtype, device=curr_pred.device)
            dense[i_x, i_y] = curr_pred
            inverse_mask = dense == 2
            dense += inverse_mask.T
            dense -= inverse_mask * 2 
            res.append(dense)
            running_ix += len(i_x)
        return res
        



class _RoisPairExpandingLayer(nn.Module):
    def __init__(self, separate=False):
        super(_RoisPairExpandingLayer,self).__init__()

    def forward(self, rois, batch_size, obj_num, separate=False):
        """
        :param rois: region of intrests list
        :param batch_size: image number in one batch
        :param obj_num: a Tensor that indicates object numbers in each image
        :return:
        """
        self._rois = torch.tensor([]).type_as(rois).float()
        indices_list = []
        roi_list = []
        for imgnum in range(obj_num.size(0)):
            begin_idx = obj_num[:imgnum].sum().item()
            if obj_num[imgnum] == 1:
                cur_rois = rois[int(begin_idx):int(begin_idx + obj_num[imgnum].item())][:, 1:5]
                cur_rois = torch.cat([((imgnum % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                      cur_rois], 1)
                self._rois = torch.cat([self._rois, cur_rois], 0)
            elif obj_num[imgnum] >1:
                cur_rois = rois[int(begin_idx):int(begin_idx + obj_num[imgnum].item())][:, 1:5]
                self_indices = [(i, i) for i in range(obj_num[imgnum])]
                cur_rois, roi_indices = self._single_image_expand(cur_rois)
                cur_rois = torch.cat([((imgnum % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                        cur_rois], 1)
                if not separate:
                    self._rois = torch.cat([self._rois, cur_rois], 0)
                else:
                    roi_list.append(cur_rois)
                    indices_list.append(self_indices + roi_indices)
                
        if not separate:    
            degenerate_boxes = self._rois[:, 3:] <= self._rois[:, 1:3]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = self._rois[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb}.",
                )
        else:
            self._rois = roi_list
        
        indices_list = [torch.tensor(i) for i in indices_list]
                
        return self._rois, indices_list


    def _single_image_expand(self, rois):
        _rois = rois
        _rois_num = _rois.size(0)
        _roi_indices = []
        for b1 in range(_rois_num):
            for b2 in range(b1+1, _rois_num):
                if b1 != b2:
                    box1 = rois[b1]
                    box2 = rois[b2]
                    tmax = torch.max(box1[2:4], box2[2:4])
                    tmin = torch.min(box1[0:2], box2[0:2])
                    unionbox = torch.cat([tmin, tmax],0)
                    unionbox = torch.reshape(unionbox, (-1, 4))
                    _rois = torch.cat([_rois, unionbox], 0)
                    _roi_indices.append((b1, b2))
        return _rois, _roi_indices


class _ObjPairLayer(nn.Module):
    def __init__(self, graph_mode=False):
        super(_ObjPairLayer, self).__init__()
        self.graph_mode = graph_mode

    def forward(self, roi_pooled_feats, batch_size, obj_num):
        """
        :param roi_pooled_feats: feature maps after roi pooling.
          The first obj_num features are single-object features.
          dim: BS*N+N(N-1) x C x W x H
        :param obj_num: object number
        :return: obj_pair_feats: dim: BS*N(N-1) x 3 x C x W x H
        """

        _paired_feats = Variable(torch.Tensor([]).type_as(roi_pooled_feats))
        for imgnum in range(obj_num.size(0)):
            if obj_num[imgnum] <= 1:
                continue
            begin_idx = (0.5 * obj_num[:imgnum].float() ** 2 + 0.5 * obj_num[:imgnum].float()).sum().item()
            cur_img_feats = roi_pooled_feats[int(begin_idx):\
                int(begin_idx + 0.5 * float(obj_num[imgnum]) ** 2 + 0.5 * float(obj_num[imgnum]))]
            cur_img_feats = self._single_image_pair(cur_img_feats, int(obj_num[imgnum]))
            _paired_feats = torch.cat([_paired_feats, cur_img_feats], 0)

        return _paired_feats

    def _single_image_pair(self, feats, objnum):
        obj_feats = feats[:objnum]
        union_feats = feats[objnum:]
        pair_feats = Variable(torch.Tensor([]).type_as(feats))

        cur_union = 0
        for o1 in range(objnum):
            for o2 in range(o1+1, objnum):
                pair_feats = torch.cat([pair_feats,
                                        torch.cat([obj_feats[o1:o1+1],
                                                   obj_feats[o2:o2+1],
                                                   union_feats[cur_union:cur_union+1]],
                                                  0).unsqueeze(0)]
                                       ,0)
                cur_union += 1

        return pair_feats

