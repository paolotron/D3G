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
import torch_geometric.utils as utils
import os.path as osp
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data import transforms as T
import detectron2.structures as structures
import imageio.v3 as iio
import pandas as pd
from data.graph_builder import *
from data.metagraspnet_synth_mapper import MetaGraspNetV2Mapper

scene_data = None
sample_metadata = None
sample_empty_metadata = None
cache_dir = osp.dirname(__file__)

def get_rgb_test_transform(img_size=(512, 512)):
    augs = T.AugmentationList([
        T.CropTransform(372, 0, 1200, 1200),
        T.Resize(img_size)
    ])
    return augs

def get_metagraspnet_dict_empty_bin(cache_dir=cache_dir):
    global sample_empty_metadata
    if sample_empty_metadata is None:
        sample_empty_metadata = pd.read_json(osp.join(cache_dir, 'sample_empty_metadata.json'))
    
    return sample_empty_metadata.to_dict('records')
                                     

def get_metagraspnet_dict_real(split='test_all', cache_dir=cache_dir):
    global scene_data
    global sample_metadata
    
    assert split in ('test_easy', 'test_medium', 'test_all', 'debug')
    
    if scene_data is None:
        with open(osp.join(cache_dir, 'scene_real_metadata.json')) as f:
            scene_data = json.load(f)
    difficulties = {
        'test_all': set([0, 1, 2, 3]),
        'test_easy': set([1]),
        'test_medium': set([2]),
        'debug': set([0, 1, 2, 3])
    }[split]
    
    scene_ix = [int(k[5:]) for k, v in scene_data['difficulty'].items()
                        if v in difficulties]
    scene_ix = set(scene_ix)
    
    if sample_metadata is None:
        sample_metadata = pd.read_json(osp.join(cache_dir, 'sample_real_metadata.json'))
    
    if split == 'debug':
        return sample_metadata[sample_metadata['scene'].isin([565])].to_dict('records')
        
    
    valid_sample = sample_metadata['bbox'].map(lambda x:len(x)) == sample_metadata['graph'].map(lambda x:len(x)) 
    sample = sample_metadata[sample_metadata['scene'].isin(scene_ix) & (sample_metadata['num_objects'] > 0) & valid_sample]
    
    return sample.to_dict('records')

DatasetCatalog.register('meta_graspnet_v2_real_test', func=lambda: get_metagraspnet_dict_real('test_all'))



class MetaGraspNetV2MapperReal(MetaGraspNetV2Mapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_on = False
        self.rgb_transform = get_rgb_test_transform(self.img_size)
        
        
    
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
            depth = depth * -1
            depth[depth == 0] = 255 
        
        # Instance Seg
        if self.mask_on:
            with np.load(npz) as file_npz:
                instance_seg = file_npz['instances_objects']
        else:
            instance_seg = None
            
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
        

if __name__ == '__main__':

    data_dict = get_metagraspnet_dict_empty_bin()
    mapper = MetaGraspNetV2Mapper(data_root='./datasets', graph_gt_type='classification', is_train=True)
    
    sample = mapper(data_dict[3])
    sample['graph_gt'].x = sample['graph_gt'].rel_gt[:, None]
    out = cls_graph_to_dense(sample['graph_gt'])
    plot_sample(sample['image'], sample['instances'].gt_boxes, out[..., 0])
    print(out)
    