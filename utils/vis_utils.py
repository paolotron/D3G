import torch_geometric
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt
import torchvision.ops as ops
import torch
import cv2
import torch_geometric.utils as utils
from utils.data_utils import cls_graph_to_dense
import numpy as np



def plot_graph(drawing, dep_graph, bbox, color=(255, 0, 0), thickness=4):
    edge_ix = dep_graph.edge_index.cpu().to(torch.int)
    for i, j in edge_ix.T:
        start = bbox[i]
        s_x, s_y = (start[0].item() + start[2].item()) // 2, (start[1].item() + start[3].item()) // 2
        end = bbox[j]
        e_x, e_y = (end[0].item() + end[2].item()) // 2, (end[1].item() + end[3].item()) // 2
        drawing = cv2.arrowedLine(drawing, (s_x, s_y), (e_x, e_y), color, thickness)
    return drawing


from collections import defaultdict
import data.metagraspnet_labels as labels
l = {int(k): v for k, v in labels.IFL_SYNSET_TO_LABEL.items()}
meta_labels = defaultdict(lambda: 'NA')
meta_labels.update(l)

def plot_sample(image, bbox, dense_graph=None, label=None):
    color = (255, 0, 0)
    thickness = 2
    if bbox.shape[0] == 0:
        drawing = image
    else:
        drawing = draw_bounding_boxes(image, bbox, labels=label,)
    drawing = drawing.permute(1, 2, 0).numpy()
    if dense_graph is not None and dense_graph.shape[0] > 1:
        edge_ix = torch.argwhere(dense_graph)
        edge_ix = edge_ix.to(torch.int).numpy()
        for i, j in edge_ix:
            start = bbox[i]
            s_x, s_y = (start[0].item() + start[2].item()) // 2, (start[1].item() + start[3].item()) // 2
            end = bbox[j]
            e_x, e_y = (end[0].item() + end[2].item()) // 2, (end[1].item() + end[3].item()) // 2
            drawing = cv2.arrowedLine(drawing, (s_x, s_y), (e_x, e_y), color, thickness)
    return drawing



def plot_errors(ax, res: dict, total='gts', del_thresh=0.01):
    assert total in ('gts', 'preds')
    missings = 'relation_from_wrong_box' if total == 'preds' else 'non_matched_gt_relations'
    name = 'predictions' if total == 'preds' else 'ground truths'
    dict_hint = """ {
        'true_positive': tp,
        'false_positive': fp,
        'correct_edge_found': correct_edge_found,
        'correct_empty': correct_empty,
        'wrong_edge_direction': wrong_edge_direction,
        'wrong_presence_of_edge': wrong_presence_of_edge,
        'missed_edge': missed_edge,
        'relation_from_wrong_box': relation_from_wrong_box,
        'non_matched_gt_relations': non_matched_gt_relations,
        'non_matched_edge_relation': non_matched_edge_relation,
        'ngt_rel': ngt_rel,
        'ngt_edge': ngt_edge,
        'all_correct': fp == 0 and tp == ngt_rel
    } """
    
    keys = ['correct_edge_found', 'correct_empty', 'wrong_edge_direction', 'missed_edge', 'wrong_presence_of_edge', missings]

    #           green      dgreen     lyellow     red       purple     grey
    colors = ['#00cc66', '#669900', '#ffcc66', '#ff3333', '#b30059', '#8c8c8c']
    values = [res[k] for k in keys]
    total = sum(values)
    if total == 0:
        return ax
    # explode = [0.1, 0, -0.1, 0, 0, 0]
    explode = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        keys, colors, values, explode = zip(*[(k, c, v, e) for k, c, v, e in zip(keys, colors, values, explode) if v / total > del_thresh])
    except:
        return ax
    keys = [k.replace('_', ' ') for k in keys]
    ax.pie(x=values,
           colors=colors,
           labels=keys,
           autopct='%1.1f%%',
           explode=explode,
           textprops={'size': 'large'}
           )
    ax.set_axis_off()
    ax.set_title(f'Distribution of {name}')
    
    return ax

def plot_only_edges(ax, res: dict, del_thresh=0.01):
    keys = ['correct_edge_found', 'wrong_edge_direction', 'missed_edge', 'non_matched_edge_relations']
    values = [res[k] for k in keys]
    total = sum(values)
    #           green     lyellow     red        grey
    colors = ['#00cc66', '#ffcc66', '#ff3333', '#8c8c8c']
    try:
        keys, colors, values = zip(*[(k, c, v) for k, c, v in zip(keys, colors, values) if v / total > del_thresh])
    except:
        return ax
    values = [res[k] for k in keys]
    total = sum(values)
    if total == 0:
        return ax
    keys = [k.replace('_', ' ') for k in keys]
    ax.pie(x=values,
           colors=colors,
           labels=keys,
           autopct='%1.1f%%',
           textprops={'size': 'large'}
           )
    ax.set_axis_off()
    ax.set_title(f'Distribution of only edges')
    return ax

def plot_pie_charts(relation_res):
    fig = plt.figure(figsize=(17, 12))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1 = plot_errors(ax1, relation_res, total='preds')
    ax2 = plot_errors(ax2, relation_res, total='gts')
    ax3 = plot_only_edges(ax3, relation_res)
    fig.tight_layout()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_output(input, output, relation_res=None, thresh=0.5):
    _detach = lambda x: x.cpu().detach()
    img = _detach(input['image']).to(torch.uint8)
    
    # End-to-End
    keep = _detach(output['instances'].scores > thresh)
    boxes = _detach(output['instances'].pred_boxes.tensor.long())[keep]
    if boxes.shape[0] > 1:
        graph = _detach(output['graph_all'][keep, :][:, keep].long())
    else:
        graph = None
    
    labels = [meta_labels[i.item()] for i in _detach(output['instances'].pred_classes[keep])]
    img_pred = plot_sample(img, boxes, graph, labels)
    
    # GT
    gt_boxes = input['instances'].gt_boxes.tensor.long()
    gt_graph = _detach(input['dense_gt'].long())
    gt_labels = [meta_labels[i.item()] for i in _detach(input['instances'].gt_classes)]
    img_gt = plot_sample(img, gt_boxes, gt_graph, gt_labels)
    
    # Only-Graph
    graph = _detach(output['graph'].long())
    img_only_graph = plot_sample(img, gt_boxes, graph, gt_labels)
    
    fig = plt.figure(figsize=(17, 12))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.imshow(img_gt)
    ax1.set_axis_off()
    ax1.title.set_text(f'GT')
    ax2.imshow(img_only_graph)
    ax2.set_axis_off()
    ax2.title.set_text(f'PRED Only Graph')
    ax3.imshow(img_pred)
    ax3.set_axis_off()
    ax3.title.set_text(f'PRED')
    # Plot Error distribution
    if relation_res is not None:
        ax4 = plot_errors(ax4, relation_res, total='preds')
        ax5 = plot_errors(ax5, relation_res, total='gts')
        ax6 = plot_only_edges(ax6, relation_res)
    fig.tight_layout()
    fig.text(.45, .25, f'ID: {input["image_id"]}')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
    