# -*- coding: utf-8 -*-
# @File: mix_up.py
# @Author: yblir
# @Time: 2022/7/3 0003 下午 11:04
# @Explain: 
# ===========================================
import os
import random

import cv2
import numpy as np


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


class MixUp:

    def __init__(self, annotations, dataset_len, target_shape):
        self.ann = annotations
        self.dataset_len = dataset_len
        self.target_shape = target_shape

    @staticmethod
    def load_image(img_path):
        img = cv2.imread(img_path)
        assert img is not None, f"file named {img_path} not found"

        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.target_shape[0] / img.shape[0], self.target_shape[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def resize_img_labels(self, index):
        '''
        将图片最长边调整到640,对应的目标框也同步调整
        '''
        img_path, target, _ = self.ann[index]
        img = self.load_image(img_path)

        r = min(self.target_shape[0] / img.shape[0],
                self.target_shape[1] / img.shape[1])
        target[:, :4] *= r
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, target

    def __call__(self, origin_img, origin_labels):
        # jit_factor = random.uniform(0.5, 1.5)
        # FLIP = random.uniform(0, 1) > 0.5
        # todo 固定比率
        jit_factor = 0.8
        FLIP = True
        # cp_index = random.randint(0, self.dataset_len - 1)
        # todo 固定mix图片
        cp_index = 5
        # 将图片最长边调整到640,对应的目标框也同步调整
        img, cp_labels = self.resize_img_labels(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((self.target_shape[0], self.target_shape[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(self.target_shape, dtype=np.uint8) * 114

        cp_scale_ratio = min(self.target_shape[0] / img.shape[0], self.target_shape[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
        : int(img.shape[0] * cp_scale_ratio),
        : int(img.shape[1] * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:  # 如果图片大于mosic,进行裁剪
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(  # 标记框位置调整
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
