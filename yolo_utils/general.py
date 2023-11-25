import os
import re
import sys

import numpy as np
from PIL import Image
import glob
from pathlib2 import Path
from datetime import datetime
import platform

import torch
from loguru import logger


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    # 这样只能处理预测一张图片的情况吧,如果批预测就不行啦
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(raw_image, target_shape, letterbox=True):
    '''
    image: pillow格式图片
    size: 处理后的尺寸
    letterbox: 是否等比例灰度填充,默认True
    '''
    iw, ih = raw_image.size
    w, h = target_shape
    if letterbox:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = raw_image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', target_shape, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = raw_image.resize((w, h), Image.BICUBIC)
    return new_image


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    '''
    获取所有类别名
    '''
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path):
    """
    这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
	作者终于良心发现，知道这个函数太复杂了！
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        sep = ''
        for n in range(2, 99999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def get_devices_info():
    """广泛用于train.py、val.py、detect.py等文件中
    用于选择模型训练的设备 并输出日志信息
    :params device: 输入的设备  device = 'cpu' or '0' or '0,1,2,3'
    :params batch_size: 一个批次的图片个数
    """
    # 如果device输入为cpu  cpu=True  device.lower(): 将device字符串全部转为小写字母
    s = f'yolo_pandora >> Python-{platform.python_version()} torch-{torch.__version__} '

    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        logger.error(f"gpu num is {num_gpu}, can't gpu train")
        sys.exit()

    if torch.cuda.is_available():
        space = ' ' * (len(s) + 1)  # 定义等长的空格
        # 满足所有条件 s加上所有显卡的信息
        for i in range(num_gpu):
            # p: 每个可用显卡的相关属性
            p = torch.cuda.get_device_properties(i)
            # 显示信息s加上每张显卡的属性信息
            s += f"{'' if i == 0 else space}|| CUDA:{i} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        # cuda不可用显示信息s就加上CPU
        logger.error(f"gpu num is {num_gpu}, but CUDA is invalid")
        sys.exit()

    # 将显示信息s加入logger日志文件中
    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe

    # 返回gpu数量,如果cuda不可用,或没有gpu,直接报错,没gpu不配炼丹~
    return num_gpu


def weights_init(net, init_type='normal', init_gain=0.02):
    '''
    初始化模型权重,这样会让训练更快收敛?
    '''

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def fuse_conv_and_bn(conv, bn):
    """在yolo.py中Model类的fuse函数中调用
    融合卷积层和BN层(测试推理使用)   Fuse convolution and batchnorm layers
    方法: 卷积层还是正常定义, 但是卷积层的参数w,b要改变   通过只改变卷积参数, 达到CONV+BN的效果
          w = w_bn * w_conv   b = w_bn * b_conv + b_bn   (可以证明)
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    https://github.com/ultralytics/yolov3/issues/807
    https://zhuanlan.zhihu.com/p/94138640
    :params conv: torch支持的卷积层
    :params bn: torch支持的bn层
    """
    fusedconv = torch.nn.Conv2d(conv.in_channels,
                                conv.out_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                groups=conv.groups,
                                bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    # w_conv: 卷积层的w参数 直接clone conv的weight即可
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # w_bn: bn层的w参数(可以自己推到公式)  torch.diag: 返回一个以input为对角线元素的2D/1D 方阵/张量?
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # w = w_bn * w_conv      torch.mm: 对两个矩阵相乘
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    # b_conv: 卷积层的b参数 如果不为None就直接读取conv.bias即可
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # b_bn: bn层的b参数(可以自己推到公式)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    #  b = w_bn * b_conv + b_bn   (w_bn not forgot)
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):  # fuse model Conv2d() + BatchNorm2d() layers
    '''
    融合conv,bn层,提高推理速度.
    '''
    from nets.network_blocks import ConvBnAct, DwConvBnAct
    logger.info('Fusing layers... ')

    for m in model.modules():
        if isinstance(m, (ConvBnAct, DwConvBnAct)) and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    return model


if __name__ == '__main__':
    a = get_devices_info()
    print(a)
