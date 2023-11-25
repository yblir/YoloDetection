# -*- coding: utf-8 -*-
# @File: calc_coco_map.py
# @Author: yblir
# @Time: 2022/7/17 0017 上午 10:56
# @Explain: 
# ===========================================
import io
import os
import sys

import cv2
import json
import random
import contextlib
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy
from yolo_utils.box_utils import xyxy2xywh, xywh2xyxy


class CocoMap:
    def __init__(self, val_json_path):
        self.coco = COCO(val_json_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]

    def convert_to_coco_format(self, outputs):
        '''
        outputs:包含nms后的预测值，img_info
        img_info:(raw_h,raw_w,coco_id)
        '''
        # 每个元素是字典,包含一个gt/预测框的信息, 一张图片大概率会对应多个字典,
        # 它们通过image_id维系,在coco数据集中,一个image_id唯一对应一张图片.
        coco_data = []
        ids = 0     # 在coco_data列表中,每个字典的得分,
        # 每次遍历取出一个batch的后处理数据
        for _, (output, img_info) in enumerate(outputs):
            batch_data = []
            # 遍历一个batch中每张图片
            for batch_id, out in enumerate(output):
                # 每张图片中预测框数量
                bboxes = out[:, 0:4]
                # 类别和对应的得分
                cls, scores = out[:, 6], out[:, 4] * out[:, 5]

                per_img_data = [None] * bboxes.shape[0]
                # 遍历每张图片上预测框.给每个预测框写入图片coco_id，得分
                for index in range(bboxes.shape[0]):
                    ids += 1
                    box = [int(b) for b in bboxes[index]]
                    # 要求是原图上的xy,wh. xy左上角坐标，不是中心点，坑死个人了~
                    coco_box = [
                        int(box[0]), int(box[1]),
                        int(box[2] - box[0]), int(box[3] - box[1])
                    ]

                    pred_data = {
                        "image_id": int(img_info[batch_id][2]),  # int
                        "category_id": self.class_ids[int(cls[index])],  # 类别
                        "bbox": coco_box,
                        "score": scores[index].item(),  # 类别得分
                        "area": coco_box[2] * coco_box[3],
                        "id": ids,  # 列表中每个字典编号

                        "iscrowd": 0,
                        "segmentation": [],
                        "ignore": 0
                    }
                    per_img_data[index] = pred_data

                batch_data.extend(per_img_data)
            coco_data.extend(batch_data)

        return coco_data

    def get_map(self, coco_data, save_dir):
        '''
        计算coco格式数据集的map
        '''
        json.dump(
            coco_data,
            open(os.path.join(save_dir, 'results.json'), 'w'),
            indent=4
        )

        coco_det = self.coco.loadRes(os.path.join(save_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()

        str_result = redirect_string.getvalue()
        (
            ap,
            ap_0_5,
            ap_7_5,
            ap_small,
            ap_medium,
            ap_large
        ) = coco_eval.stats[:6]

        print(ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result)

        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result


if __name__ == '__main__':
    coco = COCO(r'F:\GtiEE\yolo_pandora\datasets_new\coco_data\annotations\instances_val2017.json')
    coco_det = coco.loadRes(r'F:\GtiEE\yolo_pandora\runs\results.json')
    coco_eval = COCOeval(coco, coco_det, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_eval.summarize()

    str_result = redirect_string.getvalue()
    (
        ap,
        ap_0_5,
        ap_7_5,
        ap_small,
        ap_medium,
        ap_large
    ) = coco_eval.stats[:6]
    print(ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result)

