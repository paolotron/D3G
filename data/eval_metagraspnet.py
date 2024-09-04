from detectron2.evaluation import DatasetEvaluator
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
import torch
import numpy as np
from metrics.vrmn_relationship import RelationshipEval
from metrics.calculate_ap_results import  pack_instance
import metrics.oi_eval


class GraphEvaluator(DatasetEvaluator):
    
    def __init__(self, dataset_name, output_dir=None, thresh=0.3, det_only=False) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.classless = 'real' in self.dataset_name 
        self.threshold = thresh
        mkw = {'sync_on_compute': False}
        self.m_ap = MeanAveragePrecision(**mkw)
        self.m_ap_classless = MeanAveragePrecision(extended_summary=True, **mkw)
        self.iou = IntersectionOverUnion(**mkw)
        self.IoU = IntersectionOverUnion(class_metrics=True)
        self.VRMNRel = RelationshipEval(classless=self.classless, thresh=self.threshold)
        
        self.det_only = det_only
        self.ap_prep_list = []
                
    def reset(self):
        self.m_ap.reset()
        self.m_ap_classless.reset()
        self.iou.reset()
        self.VRMNRel.reset()
        self.ap_prep_list = []

    def _convert_predictions(self, d):
        d = d.get_fields()
        return {
            'boxes': d['pred_boxes'].tensor.cpu(),
            'scores': d['scores'].cpu(),
            'labels': d['pred_classes'].cpu() if not self.classless else torch.zeros_like(d['pred_classes'], device='cpu'),
        }

    def _convert_gt(self, d):
        d = d.get_fields()
        return {
            'boxes': d['gt_boxes'].tensor,
            'labels': d['gt_classes'] if not self.classless else torch.zeros_like(d['gt_classes'], device='cpu'),
        }
    
    def _convert_graph_pred(self, graph: torch.Tensor):
        
        graph = graph.bool()
        num_objs = graph.shape[0]
        if num_objs <= 1:
            return torch.zeros(0, 3), torch.zeros(0, 2)
        x, y = np.triu_indices(num_objs, 1)
        graph = torch.vstack([graph[x, y], graph[y, x]]).T
        graph = torch.concat([~(graph[:, 0]|graph[:, 1])[:, None], graph], dim=1)
        graph = graph.cpu()
        return graph, np.vstack([x, y])

    def process(self, input, output):
        assert len(input) == 1 # Adapt code if BS > 1
        gt_detection = [self._convert_gt(x['instances']) for x in input]
        predictions = [self._convert_predictions(x['instances']) for x in output]
        
        self.m_ap(preds=predictions, target=gt_detection)
        self.IoU(preds=predictions, target=gt_detection)
        
        if self.det_only:
            return

        graph_gt = [self._convert_graph_pred(x['dense_gt']) for x in input]
        graph_pred_all = [self._convert_graph_pred(x['graph_all']) for x in output]
        
        if self.save_all:
            self.save_list.append({
                'gt_bbox': gt_detection[0]['boxes'].numpy().tolist(),
                'gt_class': gt_detection[0]['labels'].numpy().tolist(),
                'pred_bbox': predictions[0]['boxes'].numpy().tolist(),
                'pred_classes': predictions[0]['labels'].numpy().tolist(),
                'pred_scores': predictions[0]['scores'].numpy().tolist(),
                'gt_graph': torch.argmax(graph_gt[0][0].float(), 1).numpy().tolist(),
                'pred_graph': graph_pred_all[0][0].cpu().numpy().tolist(),
                'image_id': input[0]['image_id']
            }) 
        
        pack = pack_instance(
            pred_bboxes=predictions[0]['boxes'].numpy(),
            pred_labels=predictions[0]['labels'].numpy(),
            pred_scores=predictions[0]['scores'].numpy(),
            pred_graph=graph_pred_all[0][0].cpu().numpy(),
            gt_bbox=gt_detection[0]['boxes'].numpy(),
            gt_label=gt_detection[0]['labels'].numpy(),
            gt_graph=torch.argmax(graph_gt[0][0].float(), 1).numpy(),
        )
        self.ap_prep_list.append(pack)
        
        if graph_gt[0][0].shape[0] == 0:
            return
        
        relation_res = self.VRMNRel(predictions, graph_pred_all, gt_detection, graph_gt)
        
        
    def evaluate(self):
        m_ap = self.m_ap.compute()
        if self.det_only:
            return {
            'eval/map': m_ap['map'].item(),
            'eval/map_50': m_ap['map_50'].item(),
            'eval/map_75': m_ap['map_75'].item(),
            }
        overall_accuracy = self.OverallAccuracy.compute()
        relation_res = self.VRMNRel.compute()
        eval_relation_res = {f'eval/{k}': v for k, v in relation_res.items()}
        rel_ap_metrics = metrics.oi_eval.eval_rel_results(self.ap_prep_list, ['background', 'rel'])
        rel_ap_metrics = {'eval/'+ k: v for k, v in rel_ap_metrics.items()}
            
        
        return {
            'eval/map': m_ap['map'].item(),
            'eval/map_50': m_ap['map_50'].item(),
            'eval/map_75': m_ap['map_75'].item(),
            **eval_relation_res,
            **rel_ap_metrics,
            'eval/overall_accuracy': overall_accuracy.item(),
        }
