import numbers
import warnings
import detectron2.data.transforms as T
import numpy as np
import torch
import torchvision.transforms as VT

import torchvision.transforms.functional as F
from fvcore.transforms import Transform


class EraseTransform(Transform):
    
    def __init__(self, x, y, h, w, v, inplace=False):
        """Erase the input Tensor Image with given value.
        This transform does not support PIL Image.

        Args:
            img (Tensor Image): Tensor image of size (C, H, W) to be erased
            i (int): i in (i,j) i.e coordinates of the upper left corner.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the erased region.
            w (int): Width of the erased region.
            v: Erasing value.
            inplace(bool, optional): For in-place operations. By default, is set False.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency
        Returns:
            ndarray: blended image(s).
        """
        img = F.erase(torch.from_numpy(img).permute(2, 0, 1), self.x, self.y, self.h, self.w, self.v, self.inplace)
        return img.permute(1, 2, 0).numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is a no-op.
        """
        return T.NoOpTransform()

         
    
class RamndomEraseTransform(T.Augmentation):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
    
    def get_transform(self, image):
        x, y, h, w, v = VT.RandomErasing.get_params(image.transpose(2, 0, 1), scale=self.scale, ratio=self.ratio)
        return EraseTransform(x, y, h, w, v, inplace=self.inplace)
    
    
def get_default_rgb_transform_train(img_size=(512, 512)):
    augs = T.AugmentationList([
        T.Resize(img_size),
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.5),
    ])
    return augs

def get_enhanced_rgb_transform_train(img_size=(512, 512)):
    return T.AugmentationList([
        T.RandomBrightness(0.7, 1.2),
        T.RandomFlip(prob=0.5),
        T.RandomSaturation(0.7, 1.2),
        # T.RandomRotation([0, 180]),
        T.MinIoURandomCrop((0.9), min_crop_size=0.7),
        T.Resize(img_size),
    ])

def get_v3_transform(img_size=(512, 512)):
    return  T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.5),
        T.RandomSaturation(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.Resize(img_size),
    ])

def get_color_transform(img_size=(512, 512)):
    return T.AugmentationList([
        T.RandomBrightness(0.7, 1.3),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomSaturation(0.7, 1.3),
        T.RandomLighting(2),
        T.RandomContrast(0.7, 1.3),
        T.Resize(img_size),
    ])

def get_resize_transform(img_size=(512, 512)):
    return T.AugmentationList([
        T.RandomApply(T.RandomRotation([0, 180])),
        T.RandomBrightness(0.9, 1.1),
        T.ResizeScale(min_scale=0.7, max_scale=1.3, target_width=img_size[0],target_height=img_size[1]),
    ])

def get_erase_transform(img_size=(512, 512)):
    return T.AugmentationList([
        T.RandomBrightness(0.7, 1.2),
        T.RandomApply(RamndomEraseTransform(), 0.5),
        T.Resize(img_size),
    ])

def get_extreme_salad_transform(img_size=(512, 512)):
    return T.AugmentationList([
        T.RandomApply(RamndomEraseTransform(), 0.3),
        T.RandomApply(T.RandomRotation([-20, 20])),
        T.RandomBrightness(0.5, 1.5),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomSaturation(0.5, 1.5),
        T.RandomLighting(1.2),
        T.RandomContrast(0.5, 1.5),
        T.MinIoURandomCrop(min_ious=(0.8, 0.9), min_crop_size=0.8),
        T.Resize(img_size),
    ])

def get_v4_transform(img_size):
    
    return T.AugmentationList([
        T.RandomSaturation(0.5, 1.5),
        T.RandomLighting(1.2),
        T.RandomContrast(0.5, 1.5),
        T.RandomContrast(0.5, 1.5),
        T.Resize(img_size),
    ])

def get_train_transform(name, img_size=(512,512)):
    return {
        'default': get_default_rgb_transform_train,
        'enhanced': get_enhanced_rgb_transform_train,
        'v3': get_v3_transform,
        'color': get_color_transform,
        'resize': get_resize_transform,
        'erase': get_erase_transform,
        'extreme': get_extreme_salad_transform,
        'v4': get_v4_transform,
    }[name](img_size=img_size)
