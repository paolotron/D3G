from .rcnn_vrmn import VMN_Head
from .rcnn_graph import GraphRCNN
from .rcnn_gheads import GraphHead
from .detr import Detr
from .detr_graph import GraphDetr
try:
    from .deformable_detr_graph import GraphDeformableDetr
    from .deformable_detr import DeformableDetr
except:
    print('Failed to load deformable Detr')