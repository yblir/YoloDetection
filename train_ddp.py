# -*- coding: utf-8 -*-
# @File: train_ddp.py
# @Author: yblir
# @Time: 2022/6/5 0005 下午 6:48
# @Explain: 
# ===========================================
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

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

from nets.yolo_body import YoloBody
from losses.yolox_loss_better import YoloXLoss
from yolo_utils.general import increment_path, get_devices_info

from datasets.data_prefetcher import DataPrefetcher
from configs.transfer import yaml_cfg
from yolo_utils import optimizers
from yolo_utils.scheduler import LRScheduler
from yolo_utils.model_ema import ModelEMA
from datasets.dataloader import load_data_loader
from yolo_utils.calc_coco_map import CocoMap

from yolo_utils.metrics import fitness
from yolo_utils.torch_utils import load_pre_model, calc_map, purify_model, save_model

np.set_printoptions(threshold=np.inf)

coco_map = CocoMap(yaml_cfg['val_json_path'])

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
# os.environ["OMP_NUM_THREADS"] = str(4)
# 初始化分布式系统 来自arcface用法
try:
    world_size = int(os.environ['WORLD_SIZE'])  # 分布式系统上所有节点上所有进程数总和, 一般有多少gpu就有多少进程数
    rank = int(os.environ['RANK'])  # 分布式系统上当前进程号,[0,word_size)
    dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    # dist.init_process_group(backend='nccl',
    #                         init_method=dist_url,
    #                         rank=rank,
    #                         world_size=world_size)
    # 使用环境变量法不会报错
    dist.init_process_group(
        'nccl' if platform.system() != 'Windows' else 'gloo'
    )
    print('000000')
except KeyError as k:
    world_size = 1
    rank = 0
    dist.init_process_group(
        backend='nccl' if platform.system() != 'Windows' else 'gloo',
        init_method='tcp://127.0.0.1:12584',
        rank=rank,
        world_size=world_size
    )

# weight decay of optimizer
weight_decay = 5e-4
# momentum of optimizer
momentum = 0.9

# epoch number used for warmup
warmup_epochs = 5

# minimum learning rate during warmup
warmup_lr = 1e-3
min_lr_ratio = 0.05
# learning rate for one image. During training, lr will multiply batchsize.
basic_lr_per_img = 0.01 / 64.0
max_epoch = 300  # 这个参数应该在配置参数,或者args中指定
no_aug_epochs = 15  # 最后15个epoch关闭数据增强


def cvtColor(image):
    # 这样只能处理预测一张图片的情况吧,如果批预测就不行啦
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


os.environ['LOCAL_RANK'] = '0'


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def warm_train(optimizer, ni, nw, epoch):
    xi = [0, nw]  # x interp
    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
    accumulate = max(
        1, np.interp(ni, xi, [1, 64 / yaml_cfg['batch_sz']]).round()
    )
    lf = one_cycle(1, yaml_cfg['lrf'], args.epochs)

    for j, x in enumerate(optimizer.param_groups):
        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        # bias的学习率从0.1下降到基准学习率lr*lf(epoch) 其他的参数学习率增加到lr*lf(epoch)
        # lf为上面设置的余弦退火的衰减函数
        x['lr'] = np.interp(
            ni, xi,
            [
                yaml_cfg['warmup_bias_lr'] if j == 2 else 0.0,
                # x['initial_lr'] * lf(epoch) todo initial_lr到底是什么东西?
                x['lr'] * lf(epoch)
            ]
        )
        if 'momentum' in x:
            x['momentum'] = np.interp(
                ni, xi, [yaml_cfg['warmup_momentum'], yaml_cfg['momentum']]
            )

    return optimizer, accumulate


def multi_train(imgsz):
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        # 下采样
        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

    return imgs


@logger.catch
def train(args):
    '''
    启用分布式进程后,会在每一个GPU上都启动一个main函数
    '''
    # 不同节点上gpu编号相同,设置当前使用的gpu编号
    # args.local_rank接受的是分布式launch自动传入的参数local_rank, 针对当前节点来说, 指每个节点上gpu编号
    # todo  新版本中,使用os.environ['LOCAL_RANK']替代分布式系统自动传入的--local_rank
    local_rank = int(os.environ['LOCAL_RANK'])
    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    data_type = torch.float16 if args.fp16 else torch.float32
    maps = np.zeros(yaml_cfg['num_classes'])  # mAP per class   # 保存每个类别map
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法,来达到优化运行效率的目的
    nbs = 64
    best_fitness = -1

    train_dir = increment_path('runs/train/exp')
    last_path = os.path.join(train_dir, 'last.pth')
    best_path = os.path.join(train_dir, 'best.pth')

    # total_batch_size,除以gpu数量才是分配到每个gpu上的数量
    accumulate = max(round(nbs / args.total_batch_size), 1)
    # 重设anchors框尺寸, yolovx使用不到
    # check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    if args.resume is True and args.pretrained is True:
        logger.error(f"resume and pretrained must only one can be True")
        sys.exit()

    # 模型=======================================================
    model = YoloBody(
        backbone_name=yaml_cfg['backbone'],
        neck_name=yaml_cfg['neck'],
        head_name=yaml_cfg['head']
    )
    if args.pretrained:
        # todo 错的离谱
        model = load_pre_model(model, yaml_cfg['model_path'])
    if args.resume:
        pass

    if rank == 0:  # 只在分布式系统第0个进程上创建记录日志文件
        logger.add(os.path.join(train_dir, 'train_record.log'))

    # SyncBatchNorm yolov7的新操作?
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to('cuda:0')
        logger.info('Using SyncBatchNorm()')
    else:
        model = model.to('cuda:0')

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank]
    )
    model.train()

    loss_func = YoloXLoss(yaml_cfg['num_classes'])

    # 数据*******************************************************************
    train_loader, val_loader, train_sampler = load_data_loader()
    # 数据集数量/batch_size, 每轮epoch要循环次数
    iters_per_epoch = len(train_loader)
    nw = max(round(yaml_cfg['warmup_epochs'] * iters_per_epoch), 1000)

    # 超参*******************************************************************
    # 反射出优化器
    optimizer = getattr(
        optimizers, yaml_cfg['optimizer']
    )(batch_size=4, model=model)

    lr_scheduler = LRScheduler(
        "yoloxwarmcos",
        (0.01 / 64.0) * yaml_cfg['batch_sz'],
        iters_per_epoch,
        total_epochs=max_epoch,
        warmup_epochs=warmup_epochs,
        warmup_lr_start=warmup_lr,
        no_aug_epochs=no_aug_epochs,
        min_lr_ratio=min_lr_ratio,
    )

    # start_epoch 有时不为0,因为有断点续训时, 会从断点epoch开始,即重设start_epoch
    start_epoch = 0

    logger.info("Training start...")
    ema = ModelEMA(model)
    ema.updates = iters_per_epoch * start_epoch

    t0 = time.time()  # 记录训练开始时间
    # 每次取出一个epoch的数据
    for epoch in range(start_epoch, args.epochs):
        # train_sampler相当于shuffle,打乱每个epoch内数据次序. ddp专用
        # epoch不同,数据次序不同.但重新训练时,如再次运行到epoch2,会和上一次train epoch2数据次序相同 !
        train_sampler.set_epoch(epoch)
        '''
        DataPrefetcher是个迭代器, 一个epoch结束后也会把迭代器耗尽,因此每个epoch都需要
        重新创建数据迭代器. 这种用法与yolox不同,因为yolox重构了数据集获取方式:InfiniteSampler
        '''
        pre_fetcher = DataPrefetcher(train_loader)
        # 数据相关
        inputs, targets = pre_fetcher.next()
        # 一个epoch中,当前运行的第几个batch
        cur_batch_id = 0
        optimizer.zero_grad()
        # 一个完整的while循环是一个epoch,在一个epoch中,每次取出一个batch的数据
        while inputs is not None:
            ni = cur_batch_id + iters_per_epoch * epoch
            cur_batch_id += 1
            # 选取较小的accumulate，optimizer以及momentum,慢慢的训练
            if ni <= nw:  # 更新的是optimizer和积累次数
                optimizer, accumulate = warm_train(optimizer, ni, nw, epoch)

            # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机
            # 选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if args.multi_scale:  # 多尺度训练,返回的是修改尺寸后的图片?
                imgs = multi_train(imgsz=None)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

            # 采用DDP训练 平均不同gpu之间的梯度,不是ddp,这行不影响结果
            loss *= int(os.environ["WORLD_SIZE"])

            scaler.scale(loss).backward()
            if ni % accumulate == 0:
                # scaler.step()首先把梯度的值unscale回来
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:  # 每次梯度回传更新ema
                    ema.update(model)

            logger.info(f"{epoch}_{cur_batch_id}:{loss}")

            # 更新学习率,每个batch_size都会更新学习率,yolox做法
            # 所以以后没必要在一个epoch结束后lr_scheduler.step()
            lr = lr_scheduler.update_lr(epoch * iters_per_epoch + cur_batch_id + 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # 获取下一次训练所需的数据
            inputs, targets = pre_fetcher.next()
            # end batch ****************************************************************

        # 每个epoch结束, 计算map,看情况保存模型
        if rank == 0:  # 只在0号gpu上进行
            # ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
            results = calc_map(
                ema.ema, val_loader, loss_func,
                conf_thres=0.3,  # nms置信度
                iou_thres=0.3  # nms阈值
            )
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results[:4]).reshape(1, -1))
            # best_fitness 初始值很小,可以保证第一次运行也可用
            if fi > best_fitness:
                best_fitness = fi
            # 保存模型
            save_model(
                args.save_per_epoch, epoch, fi, best_fitness, ema,
                optimizer, train_dir, model
            )
        # end epoch *********************************************************************

    # 所有训练结束,保存最后一个模型
    logger.info(f'\n{args.epochs - start_epoch + 1} epochs '
                f'completed in {(time.time() - t0) / 3600:.3f} hours.')

    # 去除optimizer,epoch等各种参数,提纯保存的last,best两个中间模型.
    for m in (last_path, best_path):
        if os.path.exists(m):
            if m is best_path:
                # 只处理best.pth
                purify_model(m, save_path=train_dir, use_fp16=False)  # strip optimizers

                logger.info(f'\nValidating {m}...')
                best_results = calc_map(
                    ema.ema, val_loader, loss_func,
                    conf_thres=0.3,  # nms置信度
                    iou_thres=0.3  # nms阈值
                )
        else:
            logger.error(f"path no exist : {m}")
            sys.exit()

    # callbacks.run('on_train_end', last, best, plots, epoch, results)

    torch.cuda.empty_cache()

    # return results


def make_parser():
    parser = argparse.ArgumentParser("yolo_pandora train parser")
    parser.add_argument('--weights', type=str, default='./weight/yolox5s.pt', help='initial weights path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument("--save_per_epoch", default=False, action="store_true", help="save model every epoch")
    # 分布式系统上的总batch_size
    parser.add_argument("--total-batch-size", type=int, default=4, help="total batch size for all gpu")

    # pretrained与resume互斥,只能选一个,都为True时,抛出异常
    parser.add_argument("--pretrained", default=True, action="store_true", help="pretrained training")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")

    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision training.", )

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

    return parser


if __name__ == "__main__":
    # configure_module()
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, args.name)
    # exp.merge(args.opts)  # 数据合并函数

    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    num_gpu = get_devices_info()
    train(args)
