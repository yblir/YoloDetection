# -*- coding: utf-8 -*-
# @File: yolox_head.py
# @Author: yblir
# @Time: 2022/6/20
# @Explain:
# ===========================================
import torch
from torch import nn

from configs.transfer import yaml_cfg
from nets.blocks import ConvBnAct, DwConvBnAct


class DecoupledHead(nn.Module):
    '''
    解耦头,yolox中,类别预测是单独预测的
    '''

    def __init__(self, num_classes, width=1.0, ch_in=(256, 512, 1024), dw=False, activation='silu'):
        super(DecoupledHead, self).__init__()
        Conv = DwConvBnAct if dw else ConvBnAct

        # 次序必须与原始模型次序一致
        self.cls_conv, self.box_conv, \
        self.cls_pred, self.box_pred, self.conf_pred, self.stems = [nn.ModuleList() for _ in range(6)]

        for i in range(len(ch_in)):
            self.stems.append(
                ConvBnAct(
                    int(width * ch_in[i]),
                    int(width * 256), k_size=1, stride=1, act_name=activation)
            )

            # 两次3x3卷积,提取特征,cls_conv与box_conv结构一样
            self.cls_conv.append(
                nn.Sequential(
                    Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation),
                    Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation)
                )
            )
            # 将通道数调整到类别数,使用原始的卷积获得类别预测
            self.cls_pred.append(
                nn.Conv2d(
                    in_channels=int(width * 256),
                    out_channels=num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                )
            )
            self.box_conv.append(
                nn.Sequential(
                    Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation),
                    Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation)
                )
            )
            # 框预测与置信度预测
            self.box_pred.append(
                nn.Conv2d(
                    in_channels=int(width * 256),
                    out_channels=4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                )
            )
            self.conf_pred.append(
                nn.Conv2d(
                    in_channels=int(width * 256),
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                )
            )

    def forward(self, *inputs):
        '''
        :param inputs: list,[[256,80,80],[512,40,40],[1024,20,20]]
        :return:
        '''
        outputs = []
        for i, x in enumerate(inputs):
            x = self.stems[i](x)
            cls_feat = self.cls_conv[i](x)
            cls_pred = self.cls_pred[i](cls_feat)  # num_classes,20,20

            box_feat = self.box_conv[i](x)
            box_pred = self.box_pred[i](box_feat)  # 4,20,20
            conf_pred = self.conf_pred[i](box_feat)  # 1,20,20

            # 框坐标,置信度,类别
            output = torch.cat([box_pred, conf_pred, cls_pred], dim=1)
            outputs.append(output)

        return outputs


def decoupled_head():
    return DecoupledHead(num_classes=yaml_cfg['num_classes'],
                         width=yaml_cfg['width'],
                         dw=yaml_cfg['depth_wise'])
