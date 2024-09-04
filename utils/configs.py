# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_dep_graph_config(cfg):
    """
    Add generic dependency config
    """
    cfg.INPUT.RGB = True
    cfg.INPUT.DEPTH = False

    cfg.INPUT.DEP_GRAPH = False
    cfg.INPUT.OBJ_DET = False
    cfg.INPUT.CLS_GT = False
    cfg.INPUT.GRAPH_GT_TYPE = 'classification'
    cfg.INPUT.AUGMENT = 'default'
    cfg.DATASETS.ROOT = './datasets'
    
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1
    cfg.SOLVER.GRAD_STEP = 1
    
    cfg.OUTPUT_DIR = ''
    
    cfg.TEST.GRAPH_THRESH = 0.5
    
    cfg.DATASETS.EVAL = []
        
    return cfg



def add_detr_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NUM_CLASSES = 80

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

    # TRANSFORMER
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 6
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.PRE_NORM = False

    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.GRAPH_HEAD = CN()
    cfg.MODEL.GRAPH_HEAD.NAME = 'StandardHead'
    cfg.MODEL.GRAPH_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.GRAPH_HEAD.NUM_HEADS = 1
    cfg.MODEL.GRAPH_HEAD.NUM_LAYERS = 1
    cfg.MODEL.GRAPH_HEAD.EDGE_FEATURES = 'constant_zero'
    cfg.MODEL.DETR.GRAPH_CRITEREON = 'cross'
    cfg.MODEL.DETR.GRAPH_WEIGHT = 1.0
    cfg.MODEL.DETR.FINETUNE_GHEAD = False
    
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    # Deformable Params
    cfg.MODEL.DETR.DEC_N_POINTS = 4
    cfg.MODEL.DETR.ENC_N_POINTS = 4
    cfg.MODEL.DETR.TWO_STAGE = False
    cfg.MODEL.DETR.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.DETR.POS_EMBEDDING = 'sine'
        
    return cfg