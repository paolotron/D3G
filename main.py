from collections import OrderedDict
import pathlib

from tqdm import tqdm
from utils.configs import add_dep_graph_config, add_detr_config


from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from utils.data_utils import build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
import detectron2.evaluation
import data.eval_metagraspnet
import detectron2.utils
import models
import logging
from detectron2.checkpoint import DetectionCheckpointer
import os

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from utils.train_utils import build_optimizer, EvalTestHook
from detectron2.engine import hooks
from fvcore.nn.precise_bn import get_bn_modules
import data
import os.path as osp
import json
import datetime

logger = logging.getLogger("detectron2")

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = data.get_mapper(cfg.DATASETS.TRAIN[0])(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, pin_memory=True)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        mapper = data.get_mapper(dataset_name)(cfg, is_train=False)
        dataset = get_detection_dataset_dicts(names=dataset_name)
        return build_detection_test_loader(dataset=dataset, mapper=mapper, num_workers=cfg.DATALOADER.NUM_WORKERS, )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, fast=True, save_all=False):
        det_only = cfg.MODEL.META_ARCHITECTURE in ('GeneralizedRCNN', 'Detr', 'DeformableDetr')
        
        return data.eval_metagraspnet.GraphEvaluator(dataset_name, cfg.OUTPUT_DIR, thresh=cfg.TEST.GRAPH_THRESH, det_only=det_only)
    
    @classmethod
    def test(cls, cfg, model, datasets, evaluators=None, fast=True, save_all=False):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler)
        
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(datasets) == len(evaluators), "{} != {}".format(
                len(datasets), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in tqdm(enumerate(datasets)):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, fast=fast, save_all=save_all)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'TOTAL PARAMETERS {total_params}')
        try:
            graph_head = model.detr.graph_embed
            graph_params = sum(p.numel() for p in graph_head.graph_transformer_layers.parameters())
            print(f'GRAPH PARAMETERS {graph_params}')
        except AttributeError:
            pass
        
        return build_optimizer(cfg, model)
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, datasets=cfg.DATASETS.TEST)
            return self._last_eval_results
        
        def eval_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, datasets=cfg.DATASETS.EVAL)
            return self._last_eval_results
            

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalTestHook(cfg.TEST.EVAL_PERIOD, test_function=test_and_save_results, eval_function=eval_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        
        return ret

    def build_writers(self):
        PathManager.mkdirs(self.cfg.OUTPUT_DIR)
        return [
            detectron2.utils.events.CommonMetricPrinter(self.max_iter),
            detectron2.utils.events.JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            detectron2.utils.events.TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]


def setup(args):
    cfg = get_cfg()
    cfg = add_dep_graph_config(cfg)
    if 'detr' in args.config_file: 
        cfg = add_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    if args.data_path is not None:
        cfg.DATASETS.ROOT = args.data_path
    
    opts = [i.split('=') for i in args.opts]
    opts = [x for xs in opts for x in xs]
    run_id = f"{pathlib.Path(args.config_file).name[:-1]}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    cfg.merge_from_list(opts)
    if cfg.OUTPUT_DIR == '':
        cfg.OUTPUT_DIR = f"./output/{''.join(cfg.DATASETS.TRAIN)}_out/{run_id}"
     # remove .yaml
    cfg.NAME = args.name
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        res = Trainer.test(cfg, model, datasets=cfg.DATASETS.TEST, fast=False, save_all=False)
        print(res)
        with open(osp.join(cfg.OUTPUT_DIR, 'test_results.json'), mode='w') as f:
            json.dump(res, f)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()




if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--data-path', type=str, default=None, help='root path for the dataset folder, if specified overwrites the one defined from the config file')
    parser.add_argument('--name', type=str, default=None, help='custom experiment name if needed')
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
