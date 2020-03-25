from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# import torchvision.transforms.functional as FF

'''
Data argumentation file 
modifed from 
https://github.com/ClementPinard/FlowNetPytorch


'''



'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))

        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format

        return tensor.float()


class ArrayToPILImage(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))

        img = Image.fromarray(array.astype(np.uint8))

        return img

class PILImageToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, img):
        assert(isinstance(img, Image.Image))

        array = np.asarray(img)
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)

        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        # h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        # x2 = int(round((w2 - tw) / 2.))
        # y2 = int(round((h2 - th) / 2.))
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
        # inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        # inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs,target

class myRandomResized(object):
    """
    based on RandomResizedCrop in
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
    """

    def __init__(self, expect_min_size, scale=(0.8, 1.5), interpolation=cv2.INTER_NEAREST):
        # assert (min(input_size) * min(scale) > max(expect_size))
        # one consider one decimal !!
        assert (isinstance(scale,tuple) and len(scale)==2)
        self.interpolation = interpolation
        self.scale = [ x*0.1 for x in range(int(scale[0]*10),int(scale[1])*10 )]
        self.min_size = expect_min_size

    @staticmethod
    def get_params(img, scale, min_size):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # area = img.size[0] * img.size[1]
        h, w, _ = img.shape
        for attempt in range(10):
            rand_scale_ = random.choice(scale)

            if random.random() < 0.5:
                rand_scale = rand_scale_
            else:
                rand_scale = -1.

            if min_size[0] <= rand_scale * h and min_size[1] <= rand_scale * w\
                    and rand_scale * h % 16 == 0 and rand_scale * w %16 ==0 :
                # the 16*n condition is for network architecture
                return (int(rand_scale * h),int(rand_scale * w ))

        # Fallback
        return (h, w)

    def __call__(self, inputs, tgt):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        h,w = self.get_params(inputs[0], self.scale, self.min_size)
        for i in range(len(inputs)):
            inputs[i] =  cv2.resize(inputs[i], (w,h), self.interpolation)

        tgt =  cv2.resize(tgt, (w,h), self.interpolation) #for input as h*w*1 the output is h*w
        return inputs, np.expand_dims(tgt,-1)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(min_size={0}'.format(self.min_size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)

        target = ndimage.interpolation.zoom(target, ratio, order=self.order)
        target *= ratio
        return inputs, target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th,x1: x1 + tw]
            # inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
            # inputs[2] = inputs[2][y1: y1 + th, x1: x1 + tw]

        return inputs, target[y1: y1 + th,x1: x1 + tw]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            for i in range(len(inputs)):
                inputs[i] = np.copy(np.fliplr(inputs[i]))
                # inputs[1] = np.copy(np.fliplr(inputs[1]))
                # inputs[2] = np.copy(np.fliplr(inputs[2]))

            target = np.copy(np.fliplr(target))
            # target[:,:,0] *= -1
        return inputs,target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            for i in range(len(inputs)):
                inputs[i] = np.copy(np.flipud(inputs[i]))
                # inputs[1] = np.copy(np.flipud(inputs[1]))
                # inputs[2] = np.copy(np.flipud(inputs[2]))

            target = np.copy(np.flipud(target))
            # target[:,:,1] *= -1 #for disp there is no y dim
        return inputs,target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs,target):
        applied_angle = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180

        h, w, _ = target.shape

        def rotate_flow(i,j,k):
            return -k*(j-w/2)*(diff*np.pi/180) + (1-k)*(i-h/2)*(diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] + np.sin(angle1_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target_[:,:,1]
        return inputs,target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target


class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:,:,random_order]
        inputs[1] = inputs[1][:,:,random_order]

        return inputs, target
