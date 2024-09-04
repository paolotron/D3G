import json
from typing import Any
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.config import configurable
import numpy as np
import torch
import torch
import torch_geometric.utils as utils
import torch_geometric.data as data_g
import torch_geometric as tg
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch
import cv2
import torch_geometric.utils as utils

import os.path as osp
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data import transforms as T
import detectron2.structures as structures
import imageio.v3 as iio
import pandas as pd
import tqdm

from data.augmentations import get_train_transform
from data.graph_builder import objgraph_to_objrelgraph
# from data.graph_builder import *



scene_data = None
splits = None
sample_metadata = None
cache_dir = osp.dirname(__file__)

def get_test_rgb_transform_test(img_size=(512, 512)):
    augs = T.AugmentationList([
        T.Resize(img_size)
    ])
    return augs
                                     

def get_metagraspnet_dict_synth(split='all', cache_dir=cache_dir):
    global scene_data
    global splits
    global sample_metadata
    if scene_data is None:
        with open(osp.join(cache_dir, 'scene_synt_metadata.json')) as f:
            scene_data = json.load(f)
    if splits is None:       
        with open(osp.join(cache_dir, 'splits.json')) as f:
            splits = json.load(f)
    if split == 'all':
        scene_ix = splits['train'] + splits['test'] + splits['val']
    if split.startswith('test'):
        difficulty = split.split('_')[1]
        scene_ix = splits['test']
        difficulty_ix = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        scene_difficulty = {int(k[5:]): v for k, v in scene_data['difficulty'].items()}
        scene_ix = set([ix for ix in scene_ix if scene_difficulty[ix] == difficulty_ix])
    else:
        scene_ix = splits[split]
    scene_ix = set(scene_ix)
    if sample_metadata is None:
        sample_metadata = pd.read_json(osp.join(cache_dir, 'sample_metadata.json'))
    sample = sample_metadata[sample_metadata['scene'].isin(scene_ix)]
    sample = sample[sample['num_objects'] > 1]

    # TODO REMOVE ME
    sample = sample_metadata.iloc[:1]
    
    return sample.to_dict('records')
    

DatasetCatalog.register('meta_graspnet_v2_synth_train', func=lambda: get_metagraspnet_dict_synth('train'))
DatasetCatalog.register('meta_graspnet_v2_synth_test_easy', func=lambda: get_metagraspnet_dict_synth('test_easy'))
DatasetCatalog.register('meta_graspnet_v2_synth_test_medium', func=lambda: get_metagraspnet_dict_synth('test_medium'))
DatasetCatalog.register('meta_graspnet_v2_synth_test_hard', func=lambda: get_metagraspnet_dict_synth('test_hard'))
DatasetCatalog.register('meta_graspnet_v2_synth_eval', func=lambda: get_metagraspnet_dict_synth('val'))
DatasetCatalog.register('meta_graspnet_v2_synth_all', func=lambda: get_metagraspnet_dict_synth('all'))




class MetaGraspNetV2Mapper:

    @configurable
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        graph_gt_type: str = 'relation_graph',
        depth: bool = False, 
        mask_on: bool = False,
        aug='default',
        img_size = (512, 512)
        ) -> None:
        
        assert graph_gt_type in ('relation_graph', 'classification', 'dense', 'gru_graph')
        
        self.data_root = data_root
        self.is_train = is_train
        self.graph_gt_type = graph_gt_type
        self.img_size = img_size
        self.load_depth = depth
        if self.is_train:
            self.rgb_transform = get_train_transform(aug, self.img_size)
        else:
            self.rgb_transform = get_test_rgb_transform_test(self.img_size)
        self.mask_on = mask_on
            
    @classmethod
    def from_config(cls, cfg, is_train: bool=True, ): 
        return {
            'data_root': cfg.DATASETS.ROOT,
            'is_train': is_train,
            'graph_gt_type': cfg.INPUT.GRAPH_GT_TYPE,
            'depth': cfg.INPUT.DEPTH,
            'aug': cfg.INPUT.AUGMENT,
            'mask_on': cfg.MODEL.MASK_ON
        }
    
    def __call__(self, dataset_dict) -> Any:
        
        npz = osp.join(self.data_root, dataset_dict['npz_path'])
        # h5 = osp.join(self.data_root, dataset_dict['h5_path'])
        rgb_path = osp.join(self.data_root, dataset_dict['rgb_path'])
        depth_path = osp.join(self.data_root, dataset_dict['depth_path'])
        bbox = np.array(dataset_dict['bbox'], dtype=int)
        bbox_categories = dataset_dict['bbox_categories']
        
        rgb = None
        depth = None
        
        rgb = iio.imread(rgb_path) 

        if self.load_depth:
            depth = iio.imread(depth_path)
        
        # Instance Seg
        if self.mask_on:
            with np.load(npz) as file_npz:
                instance_seg = file_npz['instances_objects']
        else:
            instance_seg = None
            
        if bbox.size != 0:
            bbox = structures.BoxMode.convert(bbox, from_mode=structures.BoxMode.XYWH_ABS, to_mode=structures.BoxMode.XYXY_ABS)
        label = torch.tensor(bbox_categories).long()
        in_aug = T.AugInput(image=rgb, boxes=bbox, sem_seg=instance_seg)
        transformation = self.rgb_transform(in_aug)
        
        if depth is not None:
            depth = transformation.apply_segmentation(depth)
        num_boxes = in_aug.boxes.shape[0]
        obj_boxes = in_aug.boxes
        obj_mask = torch.Tensor(np.array([in_aug.sem_seg == i for i in range(1, num_boxes + 1)]))
        
        adj_matrix = (torch.tensor(np.atleast_2d(dataset_dict['graph'])).T * -1).long()
        
        graph_gt = self._generate_graph_gt(adj_matrix)
        
        size = in_aug.image.shape
        image = in_aug.image
        
        if depth is not None:
            image = np.concatenate([image, depth[..., None]], axis=2)
        
        instances = structures.Instances(
                    image_size=size[:2],
                    gt_boxes=structures.Boxes(obj_boxes),
                    gt_classes=label)
        
        if self.mask_on:
            if obj_mask.shape[0] == 0:
                obj_mask = obj_mask.reshape(0, 0, 0)
            instances.set('gt_masks', structures.BitMasks(obj_mask.bool()))
        
        return {
            'width': size[0],
            'height': size[1],
            'image': torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            'instances': instances,
            'graph_gt': graph_gt,
            'dense_gt': adj_matrix,
            'image_id': dataset_dict['id'],
        }
    
    def _generate_graph_gt(self, order):
        n = order.shape[0]
        if self.graph_gt_type == 'relation_graph':
            if order.sum() > 0:
                edge_ix = utils.to_torch_coo_tensor(utils.dense_to_sparse(order)[0]).indices().float()
            else:
                edge_ix = None
            geometric_graph = data_g.Data(edge_index=edge_ix, num_nodes=n)
            graph_gt = objgraph_to_objrelgraph(obj_graph=geometric_graph)
        elif self.graph_gt_type == 'classification':
            # The Visual Manipulation RelationShip paper wants them in the form n*(n-1) where the indices are given by triu_indices
            indices = np.array(np.triu_indices(n, k=1))
            graph_gt = (order.triu() + order.T.triu() * 2)[indices[0], indices[1]].int()
        elif self.graph_gt_type == 'gru_graph':
            indices = np.array(np.triu_indices(n, k=1))
            indices = torch.from_numpy(np.concatenate([indices[[0, 1]], indices[[1, 0]]], axis=1))
            r1 = order[indices[0], indices[1]]
            r2 = order[indices[1], indices[0]]
            r0 = ((r1+r2) == 0).long()
            y = torch.vstack([r0, r1, r2]).T
            y = torch.argmax(y, dim=1)
            edges = torch.cartesian_prod(torch.arange(len(y)), torch.arange(len(y))).T
            graph_gt = tg.data.Data(edge_index=edges, num_nodes=n, rel_gt=y)
        elif self.graph_gt_type == 'dense':
            graph_gt = order
        else:
            raise NotImplemented("Type of graph GT Not implemented")
        return graph_gt
        

        