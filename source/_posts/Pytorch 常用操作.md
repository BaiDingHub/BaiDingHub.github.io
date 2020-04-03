---
title: Pytorch常用操作
date: 2020-04-03 11:00:05
tags:
 - [Pytorch]
categories: 
 - [深度学习,Pytorch]
keyword: "深度学习,Pytorch"
description: "Pytorch常用操作"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/Pytorch/Pytorch%20%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C/cover.jpg?raw=true
---





# 1.指定GPU编号

**第一种方法**

- 设置当前使用的GPU设备仅为0号设备，设备名称为 `/gpu:0`：`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- 设置当前使用的GPU设备为0,1号两个设备，名称依次为 `/gpu:0`、`/gpu:1`： `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"` ，根据顺序表示优先使用0号设备,然后使用1号设备。

   注：指定GPU的命令需要放在和神经网络相关的一系列操作的前面。



**第二种方法**

```python
device_ids = [0,1]
model = model.cuda(device_ids[0])  //将模型及参数主要放置在0号卡
if len(device_ids) > 1:
    self.model = nn.DataParallel(model, device_ids=device_ids) //使用多个GPU并行运算
```





# 2、查看模型每层输出详情

  查看模型每层输出详情

```python
from torchsummary import summary
summary(your_model, input_size=(channels, H, W))
```



# 3、梯度裁剪（Gradient Clipping）

```python
import torch.nn as nn

outputs = model(data)
loss= loss_fn(outputs, target)
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
optimizer.step()
```

`nn.utils.clip_grad_norm_` 的参数：

- **parameters** – 一个基于变量的迭代器，会进行梯度归一化
- **max_norm** – 梯度的最大范数
- **norm_type** – 规定范数的类型，默认为L2

`不椭的椭圆` 提出：梯度裁剪在某些任务上会额外消耗大量的计算时间，可移步评论区查看详情



# 4、学习率衰减

```python
import torch.optim as optim
from torch.optim import lr_scheduler

# 训练前的初始化
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)  # # 每过10个epoch，学习率乘以0.1

# 训练过程中
for n in n_epoch:
    scheduler.step()
    ...
```

可用的学习率衰减方法

- 等间隔调整学习率 StepLR

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
 
'''
等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。
##########################################################
step_size(int)- 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
last_epoch(int)- 上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
'''
```



- 按需调整学习率 MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
 
'''
按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。
##########################################################
milestones(list)- 一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]
gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
'''
```



- 指数衰减调整学习率 ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
 
'''
按指数衰减调整学习率，调整公式: lr=lr∗gamma∗∗epoch lr = lr * gamma**epoch
gamma- 学习率调整倍数的底，指数为 epoch，即 gamma**epoch
'''
```



- 余弦退火调整学习率 CosineAnnealingLR

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
 
'''
以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗T_max 为周期，在一个周期内先下降，后上升。
##########################################################
T_max(int)- 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
eta_min(float)- 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
'''
```



- 自适应调整学习率 ReduceLROnPlateau

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
 
'''
当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。
##########################################################
mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
threshold(float)- 配合 threshold_mode 使用。
cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
'''
```



- 自定义调整学习率 LambdaLR

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
 
'''
为不同参数组设定不同学习率调整策略。调整规则为，
lr=base_lr∗lmbda(self.last_epoch)
fine-tune 中十分有用，我们不仅可为不同的层设定不同的学习率，还可以为其设定不同的学习率调整策略。
##########################################################
lr_lambda(function or list)- 一个计算学习率调整倍数的函数，输入通常为 step，当有多个参数组时，设为 list。
'''
```



# 5、在不同的层使用不同的学习率

我们对模型的不同层使用不同的学习率。

还是使用这个模型作为例子：

```python
net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {}'.format(name))

# 输出：
# name: cnn.VGG_16.convolution1_1.weight
# name: cnn.VGG_16.convolution1_1.bias
# name: cnn.VGG_16.convolution1_2.weight
# name: cnn.VGG_16.convolution1_2.bias
# name: cnn.VGG_16.convolution2_1.weight
# name: cnn.VGG_16.convolution2_1.bias
# name: cnn.VGG_16.convolution2_2.weight
# name: cnn.VGG_16.convolution2_2.bias
```

对 convolution1 和 convolution2 设置不同的学习率，首先将它们分开，即放到不同的列表里：

```python
conv1_params = []
conv2_params = []

for name, parms in net.named_parameters():
    if "convolution1" in name:
        conv1_params += [parms]
    else:
        conv2_params += [parms]

# 然后在优化器中进行如下操作：
optimizer = optim.Adam(
    [
        {"params": conv1_params, 'lr': 0.01},
        {"params": conv2_params, 'lr': 0.001},
    ],
    weight_decay=1e-3,
)
```



我们将模型划分为两部分，存放到一个列表里，每部分就对应上面的一个字典，在字典里设置不同的学习率。当这两部分有相同的其他参数时，就将该参数放到列表外面作为全局参数，如上面的`weight_decay`。

也可以在列表外设置一个全局学习率，当各部分字典里设置了局部学习率时，就使用该学习率，否则就使用列表外的全局学习率。

# 6、冻结某些层的参数

在加载预训练模型的时候，我们有时想冻结前面几层，使其参数在训练过程中不发生变化。

我们需要先知道每一层的名字，通过如下代码打印：

```python
net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
```

假设前几层信息如下：

```python
name: cnn.VGG_16.convolution1_1.weight,	 grad: True
name: cnn.VGG_16.convolution1_1.bias,	 grad: True
name: cnn.VGG_16.convolution1_2.weight,	 grad: True
name: cnn.VGG_16.convolution1_2.bias,	 grad: True
name: cnn.VGG_16.convolution2_1.weight,	 grad: True
name: cnn.VGG_16.convolution2_1.bias,	 grad: True
name: cnn.VGG_16.convolution2_2.weight,	 grad: True
name: cnn.VGG_16.convolution2_2.bias,	 grad: True
```

后面的True表示该层的参数可训练，然后我们定义一个要冻结的层的列表：

```python
no_grad = [
    'cnn.VGG_16.convolution1_1.weight',
    'cnn.VGG_16.convolution1_1.bias',
    'cnn.VGG_16.convolution1_2.weight',
    'cnn.VGG_16.convolution1_2.bias'
]
```

**冻结方法**如下：

```python
net = Net.CTPN()  # 获取网络结构
for name, value in net.named_parameters():
    if name in no_grad:
        value.requires_grad = False
    else:
        value.requires_grad = True
```

冻结后我们再打印每层的信息：

```python
name: cnn.VGG_16.convolution1_1.weight,	 grad: False
name: cnn.VGG_16.convolution1_1.bias,	 grad: False
name: cnn.VGG_16.convolution1_2.weight,	 grad: False
name: cnn.VGG_16.convolution1_2.bias,	 grad: False
name: cnn.VGG_16.convolution2_1.weight,	 grad: True
name: cnn.VGG_16.convolution2_1.bias,	 grad: True
name: cnn.VGG_16.convolution2_2.weight,	 grad: True
name: cnn.VGG_16.convolution2_2.bias,	 grad: True
```

可以看到前两层的weight和bias的requires_grad都为False，表示它们不可训练。

最后在定义优化器时，只对requires_grad为True的层的参数进行更新。

```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
```
