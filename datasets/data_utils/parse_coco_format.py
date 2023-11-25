# -*- coding: utf-8 -*-
# @File: parse_coco_format.py
# @Author: yblir
# @Time: 2022/7/3 0003 下午 11:53
# @Explain: 
# ===========================================
import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO
from configs.transfer import yaml_cfg


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset:
    """
    COCO dataset class.
    """

    def __init__(self, json_path="instances_train2017.json"):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        """
        super().__init__()
        # 用于解析coco json文件,标准模块
        self.coco = COCO(os.path.join(json_path))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

    def __len__(self):
        return len(self.ids)

    def load_coco_annotations(self):
        return [self.load_ann_from_ids(_ids) for _ids in self.ids]

    def load_ann_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        target = np.zeros((num_objs, 5))
        img_path = os.path.join(yaml_cfg['train_dataset'], im_ann['file_name'])
        # todo 应该再加上图片路径
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            target[ix, 0:4] = obj["clean_bbox"]
            target[ix, 4] = cls

        # img_info在训练时用不到, 但计算map时会用到,
        # 原始图片高和宽,id_:是coco数据集自己维护的标识符,每个id匹配一张图片,
        # 同时id还匹配这张图片匹配的项目,如语义分割
        img_info = (height, width, id_)
        # x1,y1,x2,y2,c
        return img_path, target, img_info

        # r = min(self.img_size[0] / height, self.img_size[1] / width)
        # res[:, :4] *= r
        #
        # img_info = (height, width)
        # resized_info = (int(height * r), int(width * r))
        #
        # file_name = (
        #     im_ann["file_name"]
        #     if "file_name" in im_ann
        #     else "{:012}".format(id_) + ".jpg"
        # )

        # return (res, img_info, resized_info, file_name)
