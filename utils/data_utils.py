from typing import List
import torch
import numpy as np
import torch_geometric.data as data_g
import torch_geometric as tg
import itertools
import logging
import numpy as np
import operator
import pickle
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.utils.data as torchdata
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)



def cls_graph_to_dense(cls_pred: torch.Tensor, obj_num: torch.Tensor):
    res = []
    running_ix = 0
    for num in obj_num:
        if num <= 1:
            res.append(torch.tensor([]))
            continue
        i_x, i_y = np.triu_indices(n=num, k=1)
        curr_pred = cls_pred[running_ix:running_ix+len(i_x)]
        dense = torch.zeros(num, num, dtype=curr_pred.dtype, device=curr_pred.device)
        dense[i_x, i_y] = curr_pred
        inverse_mask = dense == 2
        dense += inverse_mask.T
        dense -= inverse_mask * 2 
        res.append(dense)
        running_ix += len(i_x)
    return res
        
        
def relation_graph_to_dense(graph: data_g.Data):
    num_objs = graph.is_object.sum()
    dense = torch.zeros(num_objs, num_objs, graph.x.shape[1])
    edge_index, _ = tg.utils.remove_self_loops(graph.edge_index)
    for i in range(num_objs):
        nodes, connectivity, mapping, e_mask = tg.utils.k_hop_subgraph(i, 2, edge_index, flow='source_to_target')
        connection_nodes = nodes[~graph.is_object[nodes]]
        src = connectivity[0, ::2]
        dest = connectivity[1, 1::2]
        dense[src, dest] = graph.x[connection_nodes]
    return dense


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset))
        if not isinstance(dataset, torchdata.IterableDataset)
        else None,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    **kwargs,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        **kwargs
    )
    


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch