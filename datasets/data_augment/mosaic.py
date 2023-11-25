# -*- coding: utf-8 -*-
# @File: mosaic.py
# @Author: yblir
# @Time: 2022/7/3 0003 下午 11:03
# @Explain: 
# ===========================================
import os
import random

import cv2
import numpy as np
from functools import wraps
from torch.utils.data.dataset import Dataset as torchDataset


def get_mosaic_coordinate(mosaic_index, xc, yc, w, h, target_h, target_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, target_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(target_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, target_w * 2), min(target_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    else:
        raise ValueError('mosaic index out')

    return (x1, y1, x2, y2), small_coord


class Dataset(torchDataset):
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """

    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class Mosaic:
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self, annotations, dataset_len, target_shape):
        """
        Args:
            dataset_len :
            target_shape (tuple):
            mosaic (bool): enable mosaic augmentation or not.
        """
        # super(Mosaic, self).__init__(target_shape, mosaic=True)
        self.ann = annotations
        self.dataset_len = dataset_len
        self.target_shape = target_shape

    @staticmethod
    def load_image(img_path):
        img = cv2.imread(img_path)
        assert img is not None, f"file named {img_path} not found"

        return img

    def load_resized_img(self, img):
        r = min(self.target_shape[0] / img.shape[0], self.target_shape[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index):
        img_path, target = self.ann[index]
        img = self.load_image(img_path)
        # 使最大边扩展到640
        img = self.load_resized_img(img)

        return img, target

    def __len__(self):
        return self.dataset_len

    # @Dataset.mosaic_getitem
    def __call__(self, idx):
        mosaic_labels = []
        # input_dim = self._dataset.input_dim
        target_h, target_w = self.target_shape[0], self.target_shape[1]

        # yc, xc = s, s  # mosaic center x, y
        # yc = int(random.uniform(0.5 * target_h, 1.5 * target_h))
        # xc = int(random.uniform(0.5 * target_w, 1.5 * target_w))
        # todo 固定中心
        yc = int(0.5 * target_h)
        xc = int(0.5 * target_w)
        # 取出4张图片的id
        # 3 additional image indices
        indices = [idx] + [random.randint(0, self.dataset_len - 1) for _ in range(3)]
        # todo 固定图片
        indices = [1, 2, 3, 4]
        mosaic_img = None
        for i_mosaic, index in enumerate(indices):
            # resize_img, target = self.pull_item(index)
            img_path, target, _ = self.ann[index]
            # 加载原始图片
            img = self.load_image(img_path)

            # resize图片,使最长边变为640
            img_h, img_w = img.shape[:2]  # orig hw
            scale = min(1. * target_h / img_h, 1. * target_w / img_w)
            resize_img = cv2.resize(
                img, (int(img_w * scale), int(img_h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

            # generate output mosaic image
            h, w, c = resize_img.shape[:3]
            if i_mosaic == 0:  # 生成像素值为114的灰度图
                mosaic_img = np.full((target_h * 2, target_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                i_mosaic, xc, yc, w, h, target_h, target_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = resize_img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = target.copy()
            # Normalized xywh to pixel xyxy format
            if target.size > 0:
                labels[:, 0] = scale * target[:, 0] + padw
                labels[:, 1] = scale * target[:, 1] + padh
                labels[:, 2] = scale * target[:, 2] + padw
                labels[:, 3] = scale * target[:, 3] + padh
            mosaic_labels.append(labels)

        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * target_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * target_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * target_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * target_h, out=mosaic_labels[:, 3])

        return mosaic_img, mosaic_labels
