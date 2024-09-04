import math
from models.detr_gheads import build_graph_head
# import models.detr as detr
from models.deformable_detr_modules.backbone import Joiner, build_backbone
from models.deformable_detr_modules.deformable_detr import MLP, SetCriterion
from models.deformable_detr_modules.matcher import HungarianMatcher
from models.deformable_detr_modules.position_encoding import PositionEmbeddingSine
from models.deformable_detr_modules.segmentation import PostProcessSegm
from models.deformable_detr_modules.deformable_transformer import DeformableTransformer
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.fb_misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from detectron2.structures import Boxes, Instances, BitMasks, ImageList
import torch.functional as F
import torch.nn.functional as nF
import torchvision.ops.focal_loss as focal_loss

from utils.fb_misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)


import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GRAPHDEFDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 graph_embed, aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.graph_embed = graph_embed
        self.pass_qk = False
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = nF.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        predicted_graph = self.graph_embed(hs) 
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_graph': predicted_graph[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, predicted_graph)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_graph):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
                    'pred_logits': a,
                    'pred_boxes': b,
                    'pred_graph': c
                 }
                for a, b, c in zip(outputs_class[:-1],
                                outputs_coord[:-1],
                                output_graph[:-1])]

""" 
class GRAPHDETRSegm(detr.DETRsegm):
        
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        outputs_graph = self.detr.graph_embed(memory, hs, outputs_coord)
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "pred_graph": outputs_graph[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord, outputs_graph)

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks
        return out
 """

@META_ARCH_REGISTRY.register()
class GraphDeformableDetr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        
        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        graph_critereon = cfg.MODEL.DETR.GRAPH_CRITEREON
        
        # Deformable Params
        dec_n_points = cfg.MODEL.DETR.DEC_N_POINTS
        enc_n_points = cfg.MODEL.DETR.ENC_N_POINTS
        two_stage    = cfg.MODEL.DETR.TWO_STAGE
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS
        finetune_ghead = cfg.MODEL.DETR.FINETUNE_GHEAD
        
        N_steps = hidden_dim // 2
        backbone = build_backbone(cfg)
        # backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        # backbone.num_channels = d2_backbone.num_channels

        transformer = DeformableTransformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_feature_levels=num_feature_levels,
            num_decoder_layers=dec_layers,
            return_intermediate_dec=True,
            activation='relu',
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            two_stage=two_stage,
            two_stage_num_proposals=num_queries            
        )
        
        graph_embed = build_graph_head(cfg)

        self.detr = GRAPHDEFDETR(
            backbone, transformer,
            num_classes=self.num_classes, num_queries=num_queries,
            graph_embed=graph_embed, aux_loss=deep_supervision,
            num_feature_levels=num_feature_levels
        )
        if finetune_ghead:
            for p in self.detr.parameters():
                p.requires_grad = False
            for p in self.detr.graph_embed.parameters():
                p.requires_grad = True

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality", "graph"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = GraphSetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict,
            losses=losses, graph_critereon=graph_critereon
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS

           
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_graph = [x["graph_gt"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, gt_graph)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            graph_pred = output["pred_graph"]
            results_det, results_graph_all = self.inference(box_cls, box_pred, mask_pred, graph_pred, images.image_sizes)
            result_graph = None
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_graph = [x["graph_gt"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, gt_graph)
                results_graph = self.criterion.build_dense(output, targets, results_graph_all)
            
            processed_results = []
            for results_per_image, graph_per_image, all_graph_per_image, input_per_image, image_size in zip(
                results_det, results_graph, results_graph_all, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])    
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append(
                    {'instances': r, 'graph': graph_per_image, 'graph_all': all_graph_per_image}
                )
            return processed_results
        
    def prepare_targets(self, targets, gt_graphs):
        new_targets = []
        for targets_per_image, gt_graph in zip(targets, gt_graphs):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'graph': gt_graph})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks.tensor
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, graph_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results_box = []
        results_graph = []
        
        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = nF.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        graph_pred = (graph_pred.sigmoid() > 0.5)[:, :, :, 0]

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size, graph) in enumerate(zip(
            scores, labels, box_pred, image_sizes, graph_pred
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = nF.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results_box.append(result)
            results_graph.append(graph)
        return results_box, results_graph

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images




class GraphSetCriterion(SetCriterion):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    
    def __init__(self, *args, graph_critereon='cross', **kwargs):
        super().__init__(*args, **kwargs)
        if graph_critereon == 'cross':
            self.graph_critereon = nn.BCEWithLogitsLoss()
        elif graph_critereon == 'focal':
            self.graph_critereon = focal_loss.sigmoid_focal_loss
            
    
    def build_dense(self, outputs, targets, graph):
        indices = self.matcher(outputs, targets)
        b = len(targets)
        res = []
        device = graph[0].device
        for i in range(b):
            n = targets[i]['boxes'].shape[0]
            pred_graph_res = torch.zeros(n, n, dtype=graph[i].dtype, device=device)
            src, dest = indices[i]
            src1, src2 = torch.meshgrid(src, src, indexing='ij')
            dest1, dest2 = torch.meshgrid(dest, dest, indexing='ij')
            pred_graph_res[dest1, dest2] = graph[i][src1, src2]
            res.append(pred_graph_res)
        return res
            
            
            
    
    def loss_graph(self, outputs, targets, indices, num_boxes, ):
        b = len(targets)
        
        pred_graph = outputs['pred_graph'][..., 0]
        n_pred_nodes = pred_graph.shape[1]
        dtype = pred_graph.dtype
        tot_loss = 0
        for i in range(b):
            src, dest = indices[i]
            n_gt_nodes = torch.max(dest) + 1
            transport = torch.zeros(n_pred_nodes, n_gt_nodes, device=pred_graph.device).to(dtype)
            transport[src, dest] = 1
            short_prediction = transport.T @ pred_graph[i] @ transport
            loss_graph = self.graph_critereon(short_prediction, targets[i]['graph'].float()).mean() * n_gt_nodes
            tot_loss += loss_graph
        
        tot_loss = tot_loss / b
        return {'bce_densegraph_loss': tot_loss}
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'graph': self.loss_graph
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)