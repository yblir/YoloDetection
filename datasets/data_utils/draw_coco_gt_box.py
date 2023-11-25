# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 17:50
# @Author  : yblir
# @Email   : 
# ====================================================
import os
import sys
import pycocotools.coco as coco_
import colorsys
import cv2
import numpy as np
from loguru import logger

from configs.coco_classes import COCO_CLASSES


def set_color(class_names):
    '''
    设置画框色彩函数来自yolo_utils.yolo_detect, 为了使得画coco数据集gt框
    模块可以单独使用，不依赖外部模型，把这个函数单独抽出来，放在本模块中
    Args:
        class_names: list, coco数据集80类比名字
    Returns:
    '''
    # 画框设置不同的颜色
    hsv_tuples = [
        (x / len(class_names), 1., 1.)
        for x in range(len(class_names))
    ]
    # *x: 解包(10,1.,1,.)这样的结构
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    # [(12,233,9),(...),(...)]  # 每个小元组就是一个rgb色彩值
    colors = [
        (
            int(x[0] * 255),
            int(x[1] * 255),
            int(x[2] * 255)
        )
        for x in colors
    ]
    # 打乱颜色
    np.random.seed(10101)
    np.random.shuffle(colors)
    return colors


colors = set_color(COCO_CLASSES)


def x1y1wh_x1y1x2y2(box):
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


def vis_coco_ann(img_dir, json_path, save_dir):
    # 设置文字字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    assert os.path.isfile(json_path), 'cannot find gt {}'.format(json_path)

    coco = coco_.COCO(json_path)
    images = coco.getImgIds()
    class_ids = coco.getCatIds()
    cats = coco.loadCats(class_ids)
    classes = [c["name"] for c in cats]

    num_img = len(images)

    # print('find {} samples in {}'.format(num_img, json_path))
    # print("class_ids:", class_ids)
    # print("classes:", classes)

    for index in range(num_img):
        img_id = images[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ids=ann_ids)
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']

        image_path = os.path.join(img_dir, file_name)
        assert os.path.isfile(image_path), 'cannot find {}'.format(image_path)
        img = cv2.imread(image_path)

        for ann in anns:
            bbox = [int(i) for i in ann["bbox"]]
            bbox = x1y1wh_x1y1x2y2(bbox)
            category_id = ann["category_id"]
            tracking_id = ann.get("tracking_id", None)
            label_index = class_ids.index(category_id)
            label = classes[label_index]

            # 根据当前类别画框颜色
            color = colors[label_index]

            # 画gt框
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
            )

            # show label and conf
            txt = f'{label}-{tracking_id}' if tracking_id is not None else f'{label}'
            txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            # 画标签框
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] - txt_size[1] - 2),  # x1,y1
                (bbox[0] + txt_size[0], bbox[1] - 2),  # x2,y2
                color,
                -1
            )
            # 写标签
            cv2.putText(
                img,
                txt,
                (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )
        # 保存画好框的图片
        cv2.imwrite(os.path.join(save_dir, file_name), img)
        if (index + 1) % 10 == 0:
            logger.info(f"current img process: {index + 1}")

    logger.success(f"draw img over !")


if __name__ == "__main__":
    img_dir_ = r'F:\GtiEE\yolo_pandora\datasets_new\coco_data\images\train2017'
    json_path_ = r'F:\GtiEE\yolo_pandora\datasets_new\coco_data\annotations\instances_train2017.json'
    save_dir_ = r'F:\GtiEE\yolo_pandora\runs\save_img'

    vis_coco_ann(img_dir_, json_path_, save_dir_)
