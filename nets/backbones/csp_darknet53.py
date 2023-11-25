# -*- coding: utf-8 -*-
# @File: csp_dark53.py
# @Author: yblir
# @Time: 2022/6/15 0005 下午 6:48
# @Explain:
# ===========================================
from torch import nn

from configs.transfer import yaml_cfg
from nets.blocks import ConvBnAct, DwConvBnAct, Focus, SPPBottleneck, CSPLayer


class CSPDarkNet(nn.Module):
    '''
    构建cspdarkent主干特征提取网络
    '''

    def __init__(self,
                 dep_mul, wid_mul,
                 out_features=('dark3', 'dark4', 'dark5'), dw=False, activation='silu'):
        '''
        :param dep_mul: 用于计算csp模块中,小的残差重复次数
        :param wid_mul: 用于计算初始时的通道数量, 和上面的参数,推测是用于构建不同大小权重的多个模型
        :param out_features: 最后提取的输出特征层
        :param dw: 是否使用可分离卷积
        :param activation: 激活函数
        '''
        super(CSPDarkNet, self).__init__()

        base_ch = int(wid_mul * 64)  # 初始通道数是64
        base_depth = max(round(dep_mul * 3), 1)  # round用于向下取整
        Conv = DwConvBnAct if dw else ConvBnAct

        self.out_features = out_features
        # 3,640,640->12,320,320->base_ch(64),320,320
        self.stem = Focus(ch_in=3, ch_out=base_ch,
                          k_size=3, stride=1, act_name=activation)

        # 完成卷积之后，320, 320, 64 -> 160, 160, 128
        # 完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        self.dark2 = nn.Sequential(Conv(base_ch, 2 * base_ch,
                                        k_size=3, stride=2, act_name=activation),
                                   CSPLayer(2 * base_ch, 2 * base_ch,
                                            repeat=base_depth, dw=dw, activation=activation))

        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        self.dark3 = nn.Sequential(Conv(2 * base_ch, 4 * base_ch,
                                        k_size=3, stride=2, act_name=activation),
                                   CSPLayer(4 * base_ch, 4 * base_ch,
                                            repeat=base_depth * 3, dw=dw, activation=activation))

        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        self.dark4 = nn.Sequential(Conv(4 * base_ch, 8 * base_ch,
                                        k_size=3, stride=2, act_name=activation),
                                   CSPLayer(8 * base_ch, 8 * base_ch,
                                            repeat=base_depth * 3, dw=dw, activation=activation))

        #   完成卷积之后，512,40, 40,  -> 1024,20, 20,
        #   完成SPP之后，1024,20, 20 -> 1024,20, 20,
        #   完成CSPlayer之后，1024,20, 20 -> 1024,20, 20
        self.dark5 = nn.Sequential(Conv(8 * base_ch, 16 * base_ch,
                                        k_size=3, stride=2, act_name=activation),
                                   SPPBottleneck(16 * base_ch, 16 * base_ch, k_size=(5, 9, 13)),
                                   CSPLayer(16 * base_ch, 16 * base_ch,
                                            repeat=base_depth, shortout=False, dw=dw, activation=activation))

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)

        # dark3的输出为256,80, 80，是一个有效特征层
        x = self.dark3(x)
        dark3_output = x

        # dark4的输出为512,40, 40，是一个有效特征层
        x = self.dark4(x)
        dark4_output = x

        # dark5的输出为1024,20, 20，是一个有效特征层
        x = self.dark5(x)
        dark5_output = x

        return dark3_output, dark4_output, dark5_output

    def forward_old(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x

        # dark3的输出为256,80, 80，是一个有效特征层
        x = self.dark3(x)
        outputs['dark3'] = x

        # dark4的输出为512,40, 40，是一个有效特征层
        x = self.dark4(x)
        outputs['dark4'] = x

        # dark5的输出为1024,20, 20，是一个有效特征层
        x = self.dark5(x)
        outputs['dark5'] = x

        # 构建输出层字典
        return {k: v for k, v in outputs.items() if k in self.out_features}


def csp_darknet():
    return CSPDarkNet(dep_mul=yaml_cfg['depth'],
                      wid_mul=yaml_cfg['width'],
                      dw=yaml_cfg['depth_wise'],
                      activation=yaml_cfg['activation'])
