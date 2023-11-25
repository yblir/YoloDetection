# -*- coding: utf-8 -*-
# @File: torch_utils.py
# @Author: yblir
# @Time: 2022/7/24 0024 下午 9:08
# @Explain: 该模块主要是使用torch的功能模块,主要用于train_ddp
# =======================================================
import os
import argparse
import random
import sys
import warnings
from loguru import logger
import numpy as np
import platform
import time
import math
from PIL import ImageDraw, ImageFont, Image
import cv2
import copy

np.set_printoptions(threshold=np.inf)
import torch
from torch import nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
from yolo_utils.launch import launch
from datetime import datetime

from configs.transfer import yaml_cfg
from yolo_utils.calc_coco_map import CocoMap
from yolo_utils.box_utils import decode_outputs, xywh2xyxy, xyxy2xywh, non_max_suppression, correct_boxes
from yolo_utils.general import resize_image, preprocess_input
from yolo_utils.metrics import fitness

coco_map = CocoMap(yaml_cfg['val_json_path'])

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '1'


def calc_map(
        model, val_loader, loss_func,
        conf_thres=0.3,
        iou_thres=0.3
):
    torch.cuda.empty_cache()
    model.eval()
    input_shape = np.array([640, 640])
    device = 'cuda:0'

    data_list = []
    for _, (batch_imgs, batch_boxes, batch_info) in enumerate(val_loader):
        with torch.no_grad():
            img_tensor = torch.from_numpy(batch_imgs).to(device).type(torch.float32)
            labels = [
                torch.from_numpy(ann).float().to(device)
                for ann in batch_boxes
            ]
            outputs = model(img_tensor)  # 测试阶段,不更新梯度, 直接写在no_grad()中
            # 在损失函数中会对outputs进行改变,先copy一份,用于后续计算map
            outputs_copy = copy.deepcopy(outputs)
            val_loss = loss_func(outputs, labels)

        # 解码预测框,xywh, conf和80/20个类别也sigmoid
        # (batch_size,8500,85),三个输出层拍平,
        # xywh为每个网格点相对于[640,640]尺寸归一化后的预测位置
        prediction = decode_outputs(outputs_copy, input_shape=[640, 640])

        # xyxy, obj_conf, class_conf, class_pred
        nms_detect = non_max_suppression(
            predict=prediction,
            num_classes=yaml_cfg['num_classes'],
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        # 以后的操作不再需要tensor了,转成numpy格式,释放gpu内存
        nms_detect = [item.detach().cpu().numpy() for item in nms_detect]
        # 遍历每张图片
        for i, det in enumerate(nms_detect):
            # 坐标是相对于640x640尺寸的归一化值
            det[:, :4] = xyxy2xywh(det[:, :4])
            # boxes是在原图上的坐标值,非归一化, xyxy
            boxes = correct_boxes(
                box=det[:, :4],  # xywh
                input_shape=input_shape,
                image_shape=batch_info[i][:2],  # image_shape
                letterbox=True  # 是否灰度条填充,为True,解析时会消除灰度图影响
            )
            # 最终坐标写会原来nms后的tensor数据
            nms_detect[i][:, :4] = boxes

        data_list.append((nms_detect, batch_info))
    # 所有测试数据全部跑完得到的输出并nms后，全部转换为coco格式
    # results: [1] Precision 所有类别的平均precision(最大f1时)
    #          [1] Recall 所有类别的平均recall
    #          [1] map@0.5 所有类别的平均mAP@0.5
    #          [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
    #          [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
    # maps: [80] 所有类别的mAP@0.5:0.95
    coco_data = coco_map.convert_to_coco_format(data_list)
    # todo 这个results与yolov5中不一样
    results = coco_map.get_map(coco_data, 'runs')

    return results


def purify_model(mid_model='best.pth', save_path='./purify_best.pth', use_fp16=False):
    """用在train.py模型训练完后,取出optimizer等多余部分,仅保留模型部分
    将optimizer、best_fitness、updates...从保存的模型文件f中删除
    Strip optimizer from 'f' to finalize training, optionally save as 's'
    :params mid_model: 传入的原始保存的模型文件
    :params s: 删除optimizer等变量后的模型保存的地址 dir
    :params use_fp16: 是否使用半精度模式
    """
    # x: 为加载训练的模型
    x = torch.load(mid_model, map_location=torch.device('cpu'))
    # 如果模型是ema replace model with ema
    if x.get('ema'):
        x['model'] = x['ema']
    # 以下模型训练涉及到的若干个指定变量置空
    # for k in 'optimizer', 'best_fitness', 'wandb_id', 'ema', 'updates':  # keys
    #     x[k] = None
    # x['epoch'] = -1  # 模型epoch恢复初始值-1

    if use_fp16:
        x['model'].half()  # to FP16
    # todo 模型参数全部转为不可梯度模式! 这样是否必要?
    # todo 如果只保存权重字典,是否就不需要?
    # for p in x['model'].parameters():
    #     p.requires_grad = False

    # 保存模型 x -> s/f
    # torch.save(x, save_path or mid_model)
    save_model_path = os.path.join(save_path, 'purify_best.pth')
    if isinstance(x['model'], dict):
        torch.save(x['model'], save_model_path or mid_model)
    else:
        torch.save(x['model'].state_dict(), save_model_path or mid_model)

    mb = os.path.getsize(save_model_path or mid_model) / 1E6  # filesize

    logger.info(f"Optimizer stripped from {mid_model},"
                f"{(' saved as %s,' % save_model_path) if save_model_path else ''} "
                f"{mb:.1f}MB")


def load_pre_model(model, model_path, device='cuda:0'):
    logger.info('Load weights {}.'.format(yaml_cfg['model_path']))
    try:
        model.load_state_dict(
            torch.load(model_path),
            strict=False,
            map_location=device
        )
    except Exception as e:
        logger.warning('direct load weights no success, no use 2nd method')
        model_dict = model.state_dict()
        # 此时model_list保存的都是键名,没有值
        model_list = list(model_dict)
        # 加载预训练权重,也是一个有序字典
        pre_dict = torch.load(yaml_cfg['model_path'], map_location=device)
        # 重新更新预训练模型,因为自己搭建的模型部分属性名与原始模型不同,所以不能直接加载,需要把预训练的键名替换成自己的
        # 以下是根据每层参数的shape替换原来的键名.如果构建的模型层次序或shape与原始模型不一致, 莫得法,神仙也搞不定~
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

    logger.success('load model weights success !')

    return model


def save_model(save_per_epoch, epoch, fi, best_fitness, ema, optimizer, save_dir, model):
    # Save model
    # 保存带checkpoint的模型用于inference或resuming training
    # 保存模型, 还保存了epoch, results, optimizer等信息
    # optimizer将不会在最后一轮完成后保存
    # model保存的是EMA的模型
    ckpt = {
        'epoch': epoch,
        'best_fitness': best_fitness,
        # 'model': copy.deepcopy(de_parallel(model)).half(),
        # todo 改成保存状态字典
        "model": model.state_dict(),
        'ema': copy.deepcopy(ema.ema).half(),
        'updates': ema.updates,
        'optimizer': optimizer.state_dict(),
        # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
        'date': datetime.now().isoformat()}

    # 每个epoch都保存模型
    if save_per_epoch:
        torch.save(ckpt, os.path.join(str(save_dir), f"epoch{epoch}.pth"))
    else:
        # 也会每个epoch都保存模型, 但名称为last.pth, 会覆盖之前模型.
        # 这样的last.path可用于断点续训
        torch.save(ckpt, os.path.join(str(save_dir), "last.pth"))
        if fi == best_fitness:  # 如果当前fi等于最好的fitness,则保存best.pth
            torch.save(ckpt, os.path.join(str(save_dir), "best.pth"))
    del ckpt
