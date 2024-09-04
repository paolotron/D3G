import numpy as np


def pack_instance(pred_bboxes, pred_graph, pred_scores, pred_labels, gt_bbox, gt_graph, gt_label):
    
    if pred_bboxes.shape[0] <= 1:
        pred_graph = np.zeros((0, 3), dtype=bool)
    
    pred_indices = np.vstack(np.triu_indices(pred_bboxes.shape[0], 1))
    pred_indices = pred_indices * pred_graph[:, 1] + pred_indices[[1, 0]] * pred_graph[:, 2] * ~pred_graph[:, 1]
    pred_indices = pred_indices[:, ~pred_graph[:, 0]]
    prd_scores = pred_graph.astype(np.int32)[~pred_graph[:, 0]]
    prd_scores = np.ones((prd_scores.shape[0], 2))
    
    gt_indices = np.vstack(np.triu_indices(gt_bbox.shape[0], 1))
    gt_indices = gt_indices * (gt_graph == 1).astype(np.int32) + gt_indices[[1, 0]] * (gt_graph == 2).astype(np.int32)
    gt_indices = gt_indices[:, gt_graph != 0]
    prd_gt_classes = gt_graph[gt_graph != 0]
    prd_gt_classes = np.ones((prd_gt_classes.shape[0]))

    subj_bbox = pred_bboxes[pred_indices[0]]
    sbj_scores = pred_scores[pred_indices[0]]
    subj_label = pred_labels[pred_indices[0]]

    obj_boxes = pred_bboxes[pred_indices[1]]
    obj_scores = pred_scores[pred_indices[1]]
    obj_labels = pred_labels[pred_indices[1]]

    sbj_gt_boxes = gt_bbox[gt_indices[0]]
    sbj_gt_classes = gt_label[gt_indices[0]]

    obj_gt_boxes = gt_bbox[gt_indices[1]]
    obj_gt_classes = gt_label[gt_indices[1]]


    packed_data = dict(
        sbj_boxes=subj_bbox, # N 4 box ? V
        sbj_labels=subj_label.astype(np.int32, copy=False),
        sbj_scores=sbj_scores, 
        obj_boxes=obj_boxes, # N 4 box ?
        obj_labels=obj_labels.astype(np.int32, copy=False),
        obj_scores=obj_scores,
        prd_scores=prd_scores, 
        gt_sbj_boxes=sbj_gt_boxes,
        gt_obj_boxes=obj_gt_boxes,
        gt_sbj_labels=sbj_gt_classes.astype(np.int32, copy=False),
        gt_obj_labels=obj_gt_classes.astype(np.int32, copy=False),
        gt_prd_labels=prd_gt_classes.astype(np.int32, copy=False))
    return packed_data

def pack_data(local_res):
    packed_data = []
    for ix in range(len(local_res['pred_bbox'])):
        pred_bboxes = local_res['pred_bbox'][ix]
        pred_graph = local_res['pred_graph'][ix]
        pred_scores = local_res['pred_scores'][ix]
        pred_labels = local_res['pred_classes'][ix]
        gt_bbox = local_res['gt_bbox'][ix]
        gt_graph = local_res['gt_graph'][ix]
        gt_label = local_res['gt_class'][ix]
        
        inst = pack_instance(pred_bboxes, pred_graph, pred_scores, pred_labels, gt_bbox, gt_graph, gt_label)
        packed_data.append(inst)
        
    return packed_data


