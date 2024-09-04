from collections import defaultdict
import itertools
import numpy as np
import torch.nn as nn
import torch
from torchvision.ops import box_iou

from metrics.common import filter_boxes, filter_graph


def do_rel_single_image_eval(preds_det, preds_graph, gts_det, gts_graph, iou_thresh=0.5, classless=False):
    gt_bboxes = gts_det["boxes"].cpu().numpy()
    gt_classes = gts_det["labels"].cpu().numpy()
    num_gt = gt_bboxes.shape[0]
    rel_mat_gt = np.zeros([num_gt, num_gt])
    indices = gts_graph[1]
    rel_gt = torch.argmax(gts_graph[0].float(), 1)
    rel_mat_gt[indices[0], indices[1]] = rel_gt
    rel_mat_gt = rel_mat_gt + (rel_mat_gt == 1).T *2 + (rel_mat_gt == 2).T * 1
    det_bboxes = preds_det['boxes'].cpu().numpy()
    det_labels = preds_det['labels'].cpu().numpy()
    det_rel_prob = preds_graph
    
    # total number of relationships
    ngt_rel = num_gt * (num_gt - 1) / 2
    

    # no detected rel, tp and fp is all 0
    if not det_rel_prob[0].shape[0]:
        # return 0, 0, num_gt * (num_gt - 1) /2
        res = {
            'true_positive': 0,
            'false_positive': 0,
            'all_correct': 0
        }
        return res
        
    det_rel = np.argmax(det_rel_prob[0].float(), 1)
    overlaps = box_iou(torch.from_numpy(gt_bboxes), torch.from_numpy(det_bboxes)).numpy().T
    # match bbox ground truth and detections
    match_mat = np.zeros([det_bboxes.shape[0], gt_bboxes.shape[0]])
    for i in range(det_bboxes.shape[0]):
        if classless:
            match_cand_inds = np.ones_like(det_labels[i])
        else:
            match_cand_inds = (det_labels[i] == gt_classes)
        match_cand_overlap = overlaps[i] * match_cand_inds
        # decending sort
        ovs = np.sort(match_cand_overlap, 0)
        ovs = ovs[::-1]
        inds = np.argsort(match_cand_overlap, 0)
        inds = inds[::-1]
        for ii, ov in enumerate(ovs):
            if ov > iou_thresh and np.sum(match_mat[:,inds[ii]]) == 0:
                match_mat[i, inds[ii]] = 1
                break
            elif ov < iou_thresh:
                break
    
    # true positive and false positive
    tp = 0
    fp = 0
    rel_ind = 0
    correct_edge_found = 0
    wrong_edge_direction = 0
    missed_edge = 0
    correct_empty = 0
    wrong_presence_of_edge = 0
    relation_from_wrong_box = 0
    
    for b1 in range(det_bboxes.shape[0]):
        for b2 in range(b1+1, det_bboxes.shape[0]):
            if np.sum(match_mat[b1]) > 0 and np.sum(match_mat[b2])> 0:
                b1_gt = np.argmax(match_mat[b1])
                b2_gt = np.argmax(match_mat[b2])
                rel_gt = rel_mat_gt[b1_gt, b2_gt]
                rel_pred = det_rel[rel_ind]
                
                if rel_gt != 0:
                    # WE FOUND AN EDGE
                    if rel_gt == rel_pred:
                        correct_edge_found += 1
                    elif (rel_gt != rel_pred) and rel_pred != 0:
                        wrong_edge_direction += 1
                    elif (rel_gt != rel_pred) and rel_pred == 0:
                        missed_edge += 1
                else:
                    if rel_gt == rel_pred:
                        correct_empty += 1
                    else:
                        wrong_presence_of_edge += 1
                    
                if rel_gt == rel_pred:
                    tp += 1
                else:
                    fp += 1
            else:
                relation_from_wrong_box += 1
                fp += 1
            rel_ind += 1
    
    assert fp + tp == det_bboxes.shape[0] * (det_bboxes.shape[0] - 1) / 2

    res = {
        'true_positive': tp,
        'false_positive': fp,
        'all_correct': fp == 0 and tp == ngt_rel
    }
    return res
    return tp, fp, ngt_rel

class RelationshipEval(nn.Module):
    def __init__(self, classless=False, thresh=0.5, iou_thresh=0.5,):
        self.preds = []
        super().__init__()
        self.counter = 0
        self.classless = classless
        self.threshold = thresh
        self.iou_thresh = iou_thresh
        self.results = []
        
    def reset(self):
        self.counter = 0
        self.results = []
        
    def compute_sum(self,):
        return {k: float(sum(map(lambda x: x[k], self.results))) for k in self.results[0]} 
    
    def compute(self):
        acc_res = self.compute_sum()
        
        if acc_res['true_positive'] + acc_res['false_positive'] > 0:
            o_prec = acc_res['true_positive'] / (acc_res['true_positive'] + acc_res['false_positive'])
        else:
            o_prec = 0
        o_rec = acc_res['true_positive'] / acc_res['ngt_rel']
        
        
        img_acc = acc_res['all_correct'] / len(self.results)
        
        return {
            'OP': o_prec, # Precision over all relations
            'OR': o_rec, # Recall over all relations 
            'IA': img_acc, # Complitely Solved Images 
        }
    

        
    
    def forward(self, preds_det, preds_graph, gts_det, gts_graph):
        filtered_pred_graph = [filter_graph(instances=b, graph=g, thresh=self.threshold) for b, g in zip(preds_det, preds_graph)]
        filtered_boxes = [filter_boxes(instances=b, thresh=self.threshold) for b in preds_det]
        
        res_list = []
        for p_det, p_graph, gt_det, gt_graph in zip(filtered_boxes, filtered_pred_graph, gts_det, gts_graph):
            res = do_rel_single_image_eval(p_det, p_graph, gt_det, gt_graph, iou_thresh=self.iou_thresh, classless=self.classless)
            res_list.append(res)

        self.results += res_list
        return res

        
