# -*- coding: utf-8 -*-
# @File: dataloader.py
# @Author: yblir
# @Time: 2022/7/3 0003 下午 11:25
# @Explain: 
# ===========================================
import copy
import sys

import cv2
import numpy as np
from PIL import Image
from loguru import logger
import torch
from torch.utils.data.dataset import Dataset

from configs.transfer import yaml_cfg
from datasets.data_augment import random_affine, MixUp, Mosaic
from datasets.data_utils.parse_coco_format import COCODataset
from yolo_utils.general import preprocess_input


class YoloDataset(Dataset):
    def __init__(self, data_type, phase, yaml_cfg, json_path, target_shape,
                 epoch_length=10, mosaic=True, mosaic_ratio=0.9):
        super(YoloDataset, self).__init__()
        # 判断以什么形式处理数据, 更喜欢yolov5的极简格式
        type_low = data_type.lower()
        if type_low not in ('coco', 'yolov5', 'voc'):
            logger.error(f"data_type must be in ('coco','yolov5','voc'), "
                         f"but now is {type_low}")
            sys.exit()

        if type_low == 'coco':
            self.ann = COCODataset(
                json_path=json_path
            ).load_coco_annotations()
        elif type_low == 'yolov5':
            pass
        elif type_low == 'voc':
            pass

        self.length = len(self.ann)
        self.yaml = yaml_cfg
        # self.data_aug = True if phase == 'train' else False
        self.data_aug = False
        self.mosaic = Mosaic(self.ann, self.length, target_shape)
        self.mixup = MixUp(self.ann, self.length, target_shape)

    @staticmethod
    def scale_img_box(img, box, input_shape):
        '''
        测试阶段，不适用数据增强，仅把图片和框尺寸缩放640x640
        Args:
            img:
            box:
            input_shape:

        Returns:

        '''
        # 输入的img是rgb格式
        image = Image.fromarray(img)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        resize_image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(resize_image, (dx, dy))
        # image_data = np.asarray(new_image).astype('uint8')
        image_data = np.array(new_image, dtype='float32')

        #   对真实框进行调整
        if len(box) > 0:
            # np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        # for b in box:
        #     cv2.rectangle(image_data, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        # cv2.imshow('aa', image_data)
        # cv2.waitKey()
        # box: xyxy
        return image_data, box

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 取模运算,保证任何时候index都不会超出数据集长度边界
        index = index % self.length
        # [(),(),(),...], 每个()包含一张图片的路径和未归一化的标签xyxyc
        # 此处要使用深拷贝. self.ann 元素也多为列表与数组,很容易出现浅拷贝问题, 例如元素box修改后
        # 那么self.ann也会一起改变,下一个epoch gt框坐标就是错的.
        labels = copy.deepcopy(self.ann[index])

        # if self.rand() < 0.5 and self.step_now < self.epoch_length * self.mosaic_ratio * self.length: blili条件
        # if self.enable_mosaic and random.random() < self.mosaic_prob: # yolox原版条件
        if self.yaml['use_mosaic'] and self.data_aug:  # 似乎可以有更简单的使用mosaic的随机方法
            img, box = self.mosaic(index)
        else:
            img = cv2.imread(labels[0])
            assert img is not None, f"file named {labels[0]} not found"
            box = labels[1]

        # 仿射变换
        if self.yaml['use_affine'] and self.data_aug:
            img, box = random_affine(img=img, targets=box)

        # 混淆矩阵
        if self.yaml['use_mixup'] and self.data_aug:
            img, box = self.mixup(img, box)

        # 测试阶段，不适用数据增强
        self.data_aug = False
        if not self.data_aug:
            img, box = self.scale_img_box(img, box, input_shape=(640, 640))

        # 图片归一化,并进行数值预处理!
        img = preprocess_input(img)
        # (640,640,3)->(3,640,640), BGR->RGB
        img = np.ascontiguousarray(  # img变成内存连续的数据  加快运算
            img.transpose((2, 0, 1))[::-1]
        )
        # for bbox in box:
        #    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # cv2.imshow('aa', img)
        # cv2.waitKey()

        # labels[2]是img_info=(原始图片高,宽,id(2021005))
        return img, box, labels[2]


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images, boxes, img_info = [], [], []

    for img, box, info in batch:
        images.append(img)
        boxes.append(box)
        img_info.append(info)

    images = np.array(images)

    return images, boxes, img_info


def load_data_loader():
    '''
    数据加载器
    '''
    # 数据*******************************************************************
    # Nvidia Apex,据说可以提升数据读取
    train_dataset = YoloDataset(
        data_type='coco',
        phase='train',
        yaml_cfg=yaml_cfg,
        json_path=yaml_cfg['train_json_path'],
        target_shape=(640, 640)
    )
    val_dataset = YoloDataset(
        data_type='coco',
        phase='val',
        yaml_cfg=yaml_cfg,
        json_path=yaml_cfg['val_json_path'],
        target_shape=(640, 640)
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=yaml_cfg['batch_sz'],
        # shuffle=True, # ddp训练时,shuffle与sampler冲突,不设置
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=yolo_dataset_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=yaml_cfg['batch_sz'],
        shuffle=True,  # ddp训练时,shuffle与sampler冲突,不设置
        num_workers=0,
        pin_memory=True,
        # sampler=train_sampler,
        collate_fn=yolo_dataset_collate
    )

    return train_loader, val_loader, train_sampler


if __name__ == '__main__':
    # yaml_cfg = {'use_mosaic': True,
    #             'use_affine': True,
    #             'use_mixup': True}
    # det = DetectionData('coco', yaml_cfg, (640, 640))
    # imgs, boxs = det.__getitem__(5)
    img = cv2.imread(r'F:\GtiEE\yolo_pandora\images\street.jpg')
    print(11)
