import torch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from typing import Dict, List, Optional
import torch.nn as nn


class RelCELoss(nn.Module):
    def __init__(self, reweight=(0.1, 1.0, 1.0)):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(reweight))
    
    def forward(self, x, y):
        loss = self.loss(x, y)
        preds = (torch.argmax(x, dim=1) == y.long())
        train_acc = torch.mean(preds, dtype=float)
        edge_recall = torch.mean(preds[y.long()>0], dtype=float)
        edge_recall = torch.nan_to_num(edge_recall, nan=0)
        return {'graph_bce_loss': loss, 'rel_acc': train_acc, 'edge_recall': edge_recall}
    
    
@META_ARCH_REGISTRY.register()
class GraphRCNN(GeneralizedRCNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.inference(batched_inputs, detected_instances=gt_instances)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_graphs = [x["graph_gt"] for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, gt_graph=gt_graphs)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, None)
        
        if detected_instances is not None:
            proposals_gt = [x.gt_boxes.to(images.device) for x in detected_instances]
            
            num_objs = [x.tensor.shape[0] for x in proposals_gt]    
            results, _ = self.roi_heads(images, features, proposals, None)
            proposals_pred = [x.pred_boxes for x in results]
            
            graph_pred = self.roi_heads._forward_graph(features, proposals_gt, None)
            graph_pred = self.roi_heads.pred_to_dense(graph_pred, num_objs)
            
            num_objs = [x.tensor.shape[0] for x in proposals_pred]
            graph_pred_all = self.roi_heads._forward_graph(features, proposals_pred, None)    
            graph_pred_all = self.roi_heads.pred_to_dense(graph_pred_all, num_objs)
        else:
            results, _ = self.roi_heads(images, features, proposals, None)  
            proposals_pred = [x.pred_boxes for x in results]
            num_objs = [x.tensor.shape[0] for x in proposals_pred] 
            graph_pred_all = self.roi_heads._forward_graph(features, proposals_pred, None)
            graph_pred_all = self.roi_heads.pred_to_dense(graph_pred, num_objs)
            graph_pred = [None] * len(graph_pred)
        
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            instances = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            instances = results
        for i, g, ga in zip(instances, graph_pred, graph_pred_all):
            i['graph'] = g
            i['graph_all'] = ga

        return instances