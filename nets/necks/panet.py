# -*- coding: utf-8 -*-
# @File: panet.py
# @Author: yblir
# @Time: 2022/6/15 0005 下午 6:48
# @Explain:
# ===========================================
import torch
from torch import nn

from configs.transfer import yaml_cfg
from nets.blocks import ConvBnAct, DwConvBnAct, CSPLayer


# YOLOPAFPN
class PANet(nn.Module):
    '''
    路径聚合网络,加强特征提取网络
    '''

    def __init__(self, depth=1.0, width=1.0, in_feats=('dark3', 'dark4', 'dark5'),
                 ch_in=(256, 512, 1024), dw=False, activation='silu'):
        '''
        :param depth:用于csp模块中小残差块的重复次数
        :param width:用于计算初始时的通道数量, 和上面的参数,推测是用于构建不同大小权重的多个模型
        :param in_feats:主干网络待提取的三个特征层
        :param ch_in:
        :param dw:
        :param activation:
        '''
        super(PANet, self).__init__()
        Conv = DwConvBnAct if dw else ConvBnAct

        self.in_feats = in_feats
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # todo 这些属性的名称也必须与原始模型名称一致,否则不能使用权重,
        #  当使用model_train.load_state_dict(torch.load(path,strict=False))时不再报错,但会跳过键名不一样的权重,不加载.
        # 主干特征提取网络后经过卷积,可得panet第一个输入,也是fpn段的第一次个输入. 20, 20, 1024 -> 20, 20, 512
        self.last_conv = ConvBnAct(int(width * ch_in[2]), int(width * ch_in[1]), k_size=1, stride=1,
                                   act_name=activation)
        # 40, 40, 1024 -> 40, 40, 512
        self.P4_P3_csp = CSPLayer(int(2 * width * ch_in[1]), int(width * ch_in[1]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        # 40, 40, 512 -> 40, 40, 256
        self.P4_conv = ConvBnAct(int(width * ch_in[1]), int(width * ch_in[0]), k_size=1, stride=1,
                                 act_name=activation)
        self.P3_csp = CSPLayer(int(2 * width * ch_in[0]), int(width * ch_in[0]),
                               repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        # 下采样阶段,使用卷积完成下采样过程. 有可能会用可分离卷积?
        self.down_conv1 = Conv(int(width * ch_in[0]), int(width * ch_in[0]), k_size=3, stride=2, act_name=activation)
        # 下采样阶段的csp
        self.P3_P4_csp = CSPLayer(int(2 * width * ch_in[0]), int(width * ch_in[1]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        self.down_conv2 = Conv(int(width * ch_in[1]), int(width * ch_in[1]), k_size=3, stride=2, act_name=activation)

        self.P4_P5_csp = CSPLayer(int(2 * width * ch_in[1]), int(width * ch_in[2]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)

    def forward(self, *inputs):
        # shape分别为256,80, 80|| 512,40,40 || 1024,20,20
        feat1, feat2, feat3 = inputs

        P5 = self.last_conv(feat3)  # 1024,20,20 => 512,20,20
        P5_upsample = self.upsample(P5)  # 512,20,20 => 512,40,40

        P5_P4_cat = torch.cat([P5_upsample, feat2], dim=1)  # P5的上采样与dark4的输出拼接, =>1024,40,40
        P4 = self.P4_P3_csp(P5_P4_cat)  # 拼接后还有经过csp模块,1024,40,40 => 512,40,40
        P4 = self.P4_conv(P4)  # 512,40,40 => 256,40,40

        P4_upsample = self.upsample(P4)  # 256,80,80
        P4_P3_cat = torch.cat([P4_upsample, feat1], dim=1)  # =>512,80,80

        P3_out = self.P3_csp(P4_P3_cat)  # 加强网络最终输出之一 512,80,80 => 256,80,80

        P3_down = self.down_conv1(P3_out)  # 下采样, 256,80,80 => 256,40,40
        P3_P4_cat = torch.cat([P3_down, P4], dim=1)  # 下采样阶段的拼接, =>512,40,40
        P4_out = self.P3_P4_csp(P3_P4_cat)  # 加强网络最终输出之一 512,40,40 => 512,40,40

        P4_down = self.down_conv2(P4_out)  # 512,40,40 => 512,20,20
        P4_P5_cat = torch.cat([P4_down, P5], dim=1)  # => 1024,20,20

        P5_out = self.P4_P5_csp(P4_P5_cat)  # 加强网络最终输出之一 1024,20,20 => 1024,20,20

        # shape分别为(256,80,80),(512,40,40),(1024,20,20)
        return P3_out, P4_out, P5_out


def pa_fpn():
    return PANet(depth=yaml_cfg['depth'],
                 width=yaml_cfg['width'],
                 dw=yaml_cfg['depth_wise'])
