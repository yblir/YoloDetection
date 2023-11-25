import torch
from torch import nn

# weight decay of optimizer
weight_decay = 5e-4
# momentum of optimizer
momentum = 0.9

# epoch number used for warmup
warmup_epochs = 5

# minimum learning rate during warmup
warmup_lr = 0

# learning rate for one image. During training, lr will multiply batchsize.
basic_lr_per_img = 0.01 / 64.0


# yolox优化器,给偏置,bn层,卷积层设置不同的优化方式
def yolox_optimizer(batch_size, model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    lr = warmup_lr if warmup_epochs > 0 else basic_lr_per_img * batch_size

    optimizer = torch.optim.SGD(
        pg0, lr=lr, momentum=momentum, nesterov=True
    )
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": weight_decay}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})

    return optimizer
