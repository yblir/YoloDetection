# -*- coding: utf-8 -*-
# @File: yolo_detect.py
# @Author: yblir
# @Time: 2022/7/20 0020 下午 10:41
# @Explain: 
# ===========================================
import os
import sys
import colorsys
import time
from loguru import logger
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from configs.transfer import yaml_cfg

from nets.yolo_body import YoloBody
from yolo_utils.general import resize_image
from yolo_utils.box_utils import decode_outputs, non_max_suppression


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def cvtColor(image):
    # 这样只能处理预测一张图片的情况吧,如果批预测就不行啦
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox):
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox:
        # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        # new_shape指的是宽高缩放情况
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                            box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


class YoloDetect:
    _defaults = {
        "model_path": 'weights/yolox_s.pth',
        # "classes_path": 'config/coco_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        "phi": 's',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.3,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults)
        self.class_names = yaml_cfg['classes']
        self.model_path = yaml_cfg['model_path']
        self.backbone_name = yaml_cfg['backbone']
        self.neck_name = yaml_cfg['neck']
        self.head_name = yaml_cfg['head']
        self.num_classes = yaml_cfg['num_classes']

        self.colors = self.set_color()
        self.model = self.load_model()

    def load_model(self):
        '''
        试验出一种模型键名与原始模型不同也能加载的方法,只要对应位置的shape一样就行
        '''
        model = YoloBody(
            backbone_name=self.backbone_name,
            neck_name=self.neck_name,
            head_name=self.head_name
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Load weights {}.'.format(self.model_path))
        try:
            model.load_state_dict(torch.load(model), strict=False)
        except Exception as e:
            logger.warning('direct load weights no success, no use 2nd method')
            # 键名类似:backbone.lateral_conv00.conv.weight,对应 类名.属性名.torch本身属性名
            # 类名和属性名是构建模型时自己定义的,
            model_dict = model.state_dict()
            # 此时model_list保存的都是键名,没有值
            model_list = list(model_dict)
            # 加载预训练权重,也是一个有序字典
            # todo 检测部分应该是单独的yaml文件,不和train的yaml混在一起
            pre_dict = torch.load(self.model_path, map_location=device)

            # 重新更新预训练模型,因为自己搭建的模型部分属性名与原始模型不同,
            # 所以不能直接加载,需要把预训练的键名替换成自己的
            # 以下是根据每层参数的shape替换原来的键名.如果构建的模型层次
            # 序或shape与原始模型不一致, 莫得法,神仙也搞不定~
            try:
                pre_dict = {
                    model_list[i]: v
                    for i, (k, v) in enumerate(pre_dict.items())
                    if np.shape(model_dict[model_list[i]]) == np.shape(v)
                }
            except Exception as v:
                logger.error('预训练模型每层权重shape与搭建的模型shape不一致')
                sys.exit()

            # 使用更新后的预训练权重,更新本模型权重
            model_dict.update(pre_dict)
            # 加载模型权重字典
            model.load_state_dict(model_dict)
        # 设置模型为评估模式
        model = model.eval()
        # 模型更新到gpu或cpu上
        model = model.to(device)
        logger.success(f'{self.model_path} model, and classes loaded.')

        return model

    def set_color(self):
        '''
        设置绘制的边框颜色
        :return:
        '''
        # 画框设置不同的颜色
        hsv_tuples = [
            (x / len(self.class_names), 1., 1.)
            for x in range(len(self.class_names))
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

    def detect_image(self, image):
        #   pillow格式的图片,获得输入图片的高和宽w,h
        h, w = image.size

        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        image = cvtColor(image)

        # 给图像增加灰条，实现不失真的resize，也可以直接resize进行识别
        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image
        )

        #   添加上batch_size维度
        # todo 要支持批量预测，不能使用这种方法添加维度
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype='float32')),
                (2, 0, 1)
            ),
            axis=0
        )
        # print(image_data[:100])
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda().type(torch.float32)
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            self.model = self.model.to('cpu')
            outputs = self.model(images)

        outputs = decode_outputs(outputs, self.input_shape)

        outputs[..., [0, 2]] = outputs[..., [0, 2]] / self.input_shape[1]
        outputs[..., [1, 3]] = outputs[..., [1, 3]] / self.input_shape[0]

        # [tensor,...],tensor:xyxy(640x640上的坐标), obj_conf, class_conf, class_pred
        nms_detection = non_max_suppression(
            outputs, self.num_classes, self.input_shape,
            (w, h), self.letterbox_image, conf_thres=self.confidence,
            iou_thres=self.nms_iou
        )

        results = []
        for det in nms_detection:
            det = det.cpu().numpy()
            # print('nms_det=',det[:,:4])
            # xyxy->xywh
            box_xy = (det[:, 0:2] + det[:, 2:4]) / 2
            box_wh = det[:, 2:4] - det[:, 0:2]
            # print('box_copy=',np.concatenate([box_xy,box_wh],axis=-1))
            # # 将缩放后的图片在放缩回原图
            det[:, :4] = correct_boxes(
                box_xy, box_wh,
                [640, 640], (w, h), True
            )
            # img = cv2.imread(r'I:\gitEE\yolo_pandora\datasets\coco_data\images\train2017' + '\\' + '000023.jpg')
            # for b in det[:, :4]:
            #     # cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            #     cv2.rectangle(img, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (0, 255, 0), 2)
            # # cv2.rectangle(img, (10,20), (100,200), (0, 255, 0), 2)
            # cv2.imshow('aa', img)
            # cv2.waitKey()
            # print('det=',det[:, :4])
            results.append(det)
        # if results[0] is None:
        #     return image
        # todo 要修改下 。cpu
        # top_label = np.array(results[0][:, 6].cpu(), dtype='int32')
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        #   设置字体与边框厚度
        font = ImageFont.truetype(
            font='configs/simhei.ttf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        )
        thickness = int(
            max(
                (image.size[0] + image.size[1]) // np.mean(self.input_shape),
                1
            )
        )

        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            # todo .cpu()
            # box = top_boxes[i].cpu().numpy()
            # score = top_conf[i].cpu().numpy()
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 多次画框,使得线框有合适的宽度
            for i in range(thickness+2):
                draw.rectangle(
                    (left + i, top + i, right - i, bottom - i),
                    outline=self.colors[c]
                )
            # 绘制标签框
            draw.rectangle(
                (tuple(text_origin), tuple(text_origin + label_size)),
                fill=self.colors[c]
            )
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
