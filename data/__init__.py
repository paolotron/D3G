from .metagraspnet_synth_mapper import *
from .metagraspnet_real_mapper import *

def get_mapper(name: str):

    if name.startswith('meta_graspnet_v2_synth'):
        mapper = MetaGraspNetV2Mapper
    elif  name.startswith('meta_graspnet_v2_real'):
        mapper = MetaGraspNetV2MapperReal
    else:
        mapper = MetaGraspNetV2Mapper
    return mapper