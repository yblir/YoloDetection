# -*- coding: utf-8 -*-
# @File: yolo_body.py
# @Author: yblir
# @Time: 2022/6/15 0005 下午 6:48
# @Explain:
# ===========================================
from torch import nn

# 通过反射获得模型各个模块
from nets import backbones, necks, heads

from configs.transfer import yaml_cfg


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 不能以函数的形式定义模型,因为summary时才传入input_size
class YoloBody(nn.Module):
    def __init__(self, backbone_name, neck_name, head_name):
        '''
        在yaml文件中配置网络各个模块,之后通过反射方式获得
        '''

        super(YoloBody, self).__init__()
        # 反射主干网络,getattr得到的是函数名,加()执行函数,得到返回值
        self.backbone = getattr(backbones, backbone_name)()
        # 反射颈部结构
        self.neck = getattr(necks, neck_name)()
        # 反射head网络
        self.head = getattr(heads, head_name)()

    def forward(self, inputs):
        # shape:(256,80,80),(512,40,40),(1024,20,20)
        dark3_output, dark4_output, dark5_output = self.backbone(inputs)
        # shape同上,经过neck层各模块shape不变
        p3_out, p4_out, p5_out = self.neck(dark3_output, dark4_output, dark5_output)
        # [[],[],[]], 列表有三个元素,shape分别为(bs,85,80,80),(bs,85,40,40),(bs,85,20,20)
        logit = self.head(p3_out, p4_out, p5_out)

        return logit


def weights_init(net, init_type='normal', init_gain=0.02):
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


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(
        backbone_name=yaml_cfg['backbone'],
        neck_name=yaml_cfg['neck'],
        head_name=yaml_cfg['head']
    ).to(device)
    # summary(model, input_size=(3, 640, 640))
    # torch.manual_seed(2)
    # a = torch.rand([2, 3, 640, 640]).to(device)
    # print(a[0])
    dic = model.state_dict()
    for k, v in dic.items():
      print(k)
