# -*- coding: utf-8 -*-
# @File: transfer.py
# @Author: yblir
# @Time:
# @Explain: 这个模块任务是处理yaml配置文件
# ===========================================
import os
import sys
import yaml

from loguru import logger
from pathlib2 import Path

import torch
from configs.coco_classes import COCO_CLASSES

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

param_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(param_path, 'train_param.yaml'), encoding='utf-8', errors='ignore') as f:
    yaml_cfg = yaml.safe_load(f)

depth_dic = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
width_dic = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}

phi = yaml_cfg['phi']
if phi not in ['nano', 'tiny', 's', 'm', 'l', 'x']:
    logger.error(f"phi must be in ['nano', 'tiny', 's', 'm', 'l', 'x'], but now is {phi}")
    sys.exit()

# depth:调整csp模块bottleneck(残差块)数量
# width:调整所有卷积层通道数
depth, width = depth_dic[phi], width_dic[phi]
depth_wise = True if phi == 'nano' else False

# 修改yaml_cfg参数字典值
yaml_cfg['depth'] = depth
yaml_cfg['width'] = width
yaml_cfg['depth_wise'] = depth_wise

# 当不指定设备时,优先使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if yaml_cfg['device'] is None:
    yaml_cfg['device'] = device

yaml_cfg['classes'] = COCO_CLASSES
# 为类别数量重新赋值
yaml_cfg['num_classes'] = len(yaml_cfg['classes'])

# 测试阶段......
yaml_cfg.update({'use_mosaic': True,
                 'use_affine': True,
                 'use_mixup': True})
