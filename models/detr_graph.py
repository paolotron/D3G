from models.detr_gheads import build_graph_head
import models.detr as detr
from models.detr_modules.backbone import Joiner
from models.detr_modules.detr import DETR, MLP, SetCriterion
from models.detr_modules.matcher import HungarianMatcher
from models.detr_modules.position_encoding import PositionEmbeddingSine
from models.detr_modules.segmentation import PostProcessSegm
from models.detr_modules.transformer import Transformer
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.fb_misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, Instances, BitMasks, ImageList
import torch.functional as F
import torch.nn.functional as nF
import torchvision.ops.focal_loss as focal_loss

class GRAPHDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, graph_embed, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                            DETR can detect in a single image. For COCO, we recommend 100 queries.
            graph_embed: 
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.graph_embed = graph_embed
        self.aux_loss = aux_loss
        
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
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_graph = self.graph_embed(hs)
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "pred_graph": outputs_graph[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_graph)
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
        

@META_ARCH_REGISTRY.register()
class GraphDetr(nn.Module):
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
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        graph_critereon = cfg.MODEL.DETR.GRAPH_CRITEREON
        finetune_ghead = cfg.MODEL.DETR.FINETUNE_GHEAD

        N_steps = hidden_dim // 2
        d2_backbone = detr.MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        
        graph_embed = build_graph_head(cfg)
            
        self.detr = GRAPHDETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, graph_embed=graph_embed, aux_loss=deep_supervision
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
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses, graph_critereon=graph_critereon,
            graph_weight=cfg.MODEL.DETR.GRAPH_WEIGHT
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
    
    def __init__(self, *args, graph_critereon='cross', graph_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if graph_critereon == 'cross':
            self.graph_critereon = nn.BCEWithLogitsLoss()
        elif graph_critereon == 'focal':
            self.graph_critereon = focal_loss.sigmoid_focal_loss
        self.graph_weight = graph_weight
            
    
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
            # loss_graph = self.graph_critereon(short_prediction, targets[i]['graph'].float()).mean()
            # long_gt = transport @ targets[i]['graph'].to(dtype) @ transport.T
            loss_graph = self.graph_critereon(short_prediction, targets[i]['graph'].float()).mean() * n_gt_nodes
            tot_loss += loss_graph
        
        tot_loss = tot_loss / b * self.graph_weight
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