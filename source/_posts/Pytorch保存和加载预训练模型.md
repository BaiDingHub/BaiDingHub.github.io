---
title: Pytorch保存和加载预训练模型
date: 2020-04-03 11:01:05
tags:
 - [Pytorch]
categories: 
 - [深度学习,Pytorch]
keyword: "深度学习,Pytorch,预训练模型"
description: "Pytorch保存和加载预训练模型"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/Pytorch/Pytorch%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/cover.jpg?raw=true
---






# 预训练模型使用的场景

---
>声明：该部分有部分参考，若有侵权，请及时告知

 简单来说，预训练模型(pre-trained model)是前人为了解决类似问题所创造出来的模型。你在解决问题的时候，不用从零开始训练一个新模型，可以从在类似问题中训练过的模型入手。

****场景一**：数据集小，数据相似度高(与pre-trained model的训练数据相比而言)**
在这种情况下，因为数据与预训练模型的训练数据相似度很高，因此我们不需要重新训练模型。我们只需要将输出层改制成符合问题情境下的结构就好。

我们使用预处理模型作为模式提取器。

比如说我们使用在ImageNet上训练的模型来辨认一组新照片中的小猫小狗。在这里，需要被辨认的图片与ImageNet库中的图片类似，但是我们的输出结果中只需要两项——猫或者狗。

在这个例子中，我们需要做的就是把dense layer和最终softmax layer的输出从1000个类别改为2个类别。

****场景二**：数据集小，数据相似度不高**

在这种情况下，我们可以冻结预训练模型中的前k个层中的权重，然后重新训练后面的n-k个层，当然最后一层也需要根据相应的输出格式来进行修改。

因为数据的相似度不高，重新训练的过程就变得非常关键。而新数据集大小的不足，则是通过冻结预训练模型的前k层进行弥补。

**场景三：数据集大，数据相似度不高**

在这种情况下，因为我们有一个很大的数据集，所以神经网络的训练过程将会比较有效率。然而，因为实际数据与预训练模型的训练数据之间存在很大差异，采用预训练模型将不会是一种高效的方式。

因此最好的方法还是将预处理模型中的权重全都初始化后在新数据集的基础上重头开始训练。

**场景四：数据集大，数据相似度高**

这就是最理想的情况，采用预训练模型会变得非常高效。最好的运用方式是保持模型原有的结构和初始权重不变，随后在新数据集的基础上重新训练。


# 预训练模型的方法
---
**特征提取**

我们可以将预训练模型当做特征提取装置来使用。具体的做法是，将输出层去掉，然后将剩下的整个网络当做一个固定的特征提取机，从而应用到新的数据集中。

**采用预训练模型的结构**

我们还可以采用预训练模型的结构，但先将所有的权重随机化，然后依据自己的数据集进行训练。

**训练特定层，冻结其他层**

另一种使用预训练模型的方法是对它进行部分的训练。具体的做法是，将模型起始的一些层的权重保持不变，重新训练后面的层，得到新的权重。在这个过程中，我们可以多次进行尝试，从而能够依据结果找到frozen layers和retrain layers之间的最佳搭配。

如何使用与训练模型，是由数据集大小和新旧数据集(预训练的数据集和我们要解决的数据集)之间数据的相似度来决定的。

# 实现预训练模型的加载（pytorch）
---
先附上[pytorch官方中文文档](https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-models/)、[torchvision的github地址](https://github.com/pytorch/vision/tree/master/torchvision)


### 直接加载预训练模型

```
import torchvision.models as models

model = models.resnet101(pretrained=True)
```

### 修改某一层

```
import torchvision.models as models
 
model = models.resnet101(pretrained=True)

model.fc = nn.Linear(2048, 120)  #120为样本分类数目,修改最后的分类的全连接层
model.conv1 = nn.Conv2d(3, 64,kernel_size=5, stride=2, padding=3, bias=False)   #修改中间层
```

### 加载部分预训练模型

```
#加载model，model是自己定义好的模型
resnet50 = models.resnet50(pretrained=True) 
model =Net(...) 
 
#读取参数 
pretrained_dict =resnet50.state_dict() 
model_dict = model.state_dict() 
 
#将pretrained_dict里不属于model_dict的键剔除掉 
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
 
# 更新现有的model_dict 
model_dict.update(pretrained_dict) 
 
# 加载我们真正需要的state_dict 
model.load_state_dict(model_dict)  
```



### 保存加载模型基本用法



**1、保存加载整个模型**（不推荐）

保存整个网络模型（网络结构+权重参数）。

```python
torch.save(model, 'net.pkl')
```

直接加载整个网络模型（可能比较耗时）。

```python
model = torch.load('net.pkl')
```



**2、只保存加载模型参数**（推荐）

只保存模型的权重参数（速度快，占内存少）。

```python
torch.save(model.state_dict(), 'net_params.pkl')
```

因为我们只保存了模型的参数，所以需要先定义一个网络对象，然后再加载模型参数。

```python
# 构建一个网络结构
model = ClassNet()
# 将模型参数加载到新模型中
state_dict = torch.load('net_params.pkl')
model.load_state_dict(state_dict)
```





### 保存加载自定义模型



上面保存加载的 `net.pkl` 其实一个字典，通常包含如下内容：

1. **网络结构**：输入尺寸、输出尺寸以及隐藏层信息，以便能够在加载时重建模型。
2. **模型的权重参数**：包含各网络层训练后的可学习参数，可以在模型实例上调用 `state_dict()`方法来获取，比如前面介绍只保存模型权重参数时用到的 `model.state_dict()`。
3. **优化器参数**：有时保存模型的参数需要稍后接着训练，那么就必须保存优化器的状态和所其使用的超参数，也是在优化器实例上调用 `state_dict()` 方法来获取这些参数。
4. 其他信息：有时我们需要保存一些其他的信息，比如 `epoch`，`batch_size` 等超参数。



知道了这些，那么我们就可以自定义需要保存的内容，比如：

```python
# saving a checkpoint assuming the network class named ClassNet
checkpoint = {'model': ClassNet(),
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': epoch}

torch.save(checkpoint, 'checkpoint.pkl')
```

上面的 checkpoint 是个字典，里面有4个键值对，分别表示网络模型的不同信息。



然后我们要加载上面保存的自定义的模型：

```python
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = TheOptimizerClass()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    
    return model
    
model = load_checkpoint('checkpoint.pkl')
```





### 跨设备保存加载模型



1、在 CPU 上加载在 GPU 上训练并保存的模型（Save on GPU, Load on CPU）：

```python
device = torch.device('cpu')
model = TheModelClass()
# Load all tensors onto the CPU device
model.load_state_dict(torch.load('net_params.pkl', map_location=device))
```

`map_location`：a function, torch.device, string or a dict specifying how to remap storage locations

令 `torch.load()` 函数的 `map_location` 参数等于 `torch.device('cpu')` 即可。 这里令 `map_location` 参数等于 `'cpu'` 也同样可以。



2、在 GPU 上加载在 GPU 上训练并保存的模型（Save on GPU, Load on GPU）：

```python
device = torch.device("cuda")
model = TheModelClass()
model.load_state_dict(torch.load('net_params.pkl'))
model.to(device)
```

在这里使用 `map_location` 参数不起作用，要使用 `model.to(torch.device("cuda"))` 将模型转换为CUDA优化的模型。

还需要对将要输入模型的数据调用 `data = data.to(device)`，即将数据从CPU转移到GPU。请注意，调用 `my_tensor.to(device)` 会返回一个 `my_tensor` 在 GPU 上的副本，它不会覆盖 `my_tensor`。因此需要手动覆盖张量：`my_tensor = my_tensor.to(device)`。



3、在 GPU 上加载在 GPU 上训练并保存的模型（Save on CPU, Load on GPU）

```python
device = torch.device("cuda")
model = TheModelClass()
model.load_state_dict(torch.load('net_params.pkl', map_location="cuda:0"))
model.to(device)
```

当加载包含GPU tensors的模型时，这些tensors 会被默认加载到GPU上，不过是同一个GPU设备。

当有多个GPU设备时，可以通过将 `map_location` 设定为 `*cuda:device_id*` 来指定使用哪一个GPU设备，上面例子是指定编号为0的GPU设备。

其实也可以将 `torch.device("cuda")` 改为 `torch.device("cuda:0")` 来指定编号为0的GPU设备。

最后调用 `model.to(torch.device('cuda'))` 来将模型的tensors转换为 CUDA tensors。



下面是PyTorch官方文档上的用法，可以进行参考：

```python3
>>> torch.load('tensors.pt')
# Load all tensors onto the CPU
>>> torch.load('tensors.pt', map_location=torch.device('cpu'))
# Load all tensors onto the CPU, using a function
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# Map tensors from GPU 1 to GPU 0
>>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
```



### CUDA 的用法

在PyTorch中和GPU相关的几个函数：

```text
import torch

# 判断cuda是否可用；
print(torch.cuda.is_available())

# 获取gpu数量；
print(torch.cuda.device_count())

# 获取gpu名字；
print(torch.cuda.get_device_name(0))

# 返回当前gpu设备索引，默认从0开始；
print(torch.cuda.current_device())
```



有时我们需要把数据和模型从cpu移到gpu中，有以下两种方法：

```text
use_cuda = torch.cuda.is_available()

# 方法一：
if use_cuda:
    data = data.cuda()
    model.cuda()

# 方法二：
device = torch.device("cuda" if use_cuda else "cpu")
data = data.to(device)
model.to(device)
```



个人比较习惯第二种方法，可以少一个 if 语句。而且该方法还可以通过设备号指定使用哪个GPU设备，比如使用0号设备：

```
device = torch.device("cuda:0" if use_cuda else "cpu")
```

**参考**：
[https://zhuanlan.zhihu.com/p/73893187]