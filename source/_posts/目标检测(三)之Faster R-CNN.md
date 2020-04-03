---
title: 目标检测（三）之Faster R-CNN
date: 2020-04-03 12:03:05
tags:
 - [深度基础知识]
 - [目标检测]
categories: 
 - [深度学习,目标检测]
keyword: "深度学习,目标检测,Faster R-CNN"
description: "目标检测(三)之Faster R-CNN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(%E4%B8%89)%E4%B9%8BFaster%20R-CNN/cover.jpg?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息
## &emsp;&emsp;1.论文题目
&emsp;&emsp;&emsp;&emsp;**Faster R-CNN**
## &emsp;&emsp;2.论文时间
&emsp;&emsp;&emsp;&emsp;**2015年**
## &emsp;&emsp;3.论文文献
&emsp;&emsp;&emsp;&emsp; [论文文献](https://arxiv.org/abs/1506.01497)
## &emsp;&emsp;4.论文源码
&emsp;&emsp;&emsp;&emsp;  暂无


# 二、论文背景及简介

&emsp;&emsp;&emsp;&emsp;在最早RCNN工作时，限制其运行速度的关键在于其将使用selective search生成的ROI，全部送入卷及网络进行卷积，速度很慢。Fast RCNN对其进行了优化，使用CNN提取feature map，并获得由selective search生成的ROI在feature map上的映射，将feature送入之后的卷基层进行处理，这极大地缩短了运行时间。而Faster RCNN从ROI生成的方法来进行优化，使用RPN来替换selective search，来生成ROI，这从另一个角度降低了运行时间。
&emsp;&emsp;&emsp;&emsp;论文表示，Faster RCNN　在test阶段，198ms 一张图片；Fast RCNN　320ms 一张图片

# 三、知识储备
## &emsp;&emsp; 1、anchor
&emsp;&emsp;&emsp;&emsp; `anchor`实际上就是一个在原始图片空间的`Region Proposal`，其有大小(宽×高）、有宽高比(`aspect rations`)，作者使用`anchor`这个词来代指`feature map`上可能对应的这样的一个`Region Proposal`。作者对每一块区域（３×３,即RPN的滑动窗口大小），假定存在９个`anchor`（大小分别为128、256、512，宽高比为1:1、1:2、2:1，这样３＊３中anchor）。
## &emsp;&emsp; 2、RPN（Region Proposal Networks)
&emsp;&emsp;&emsp;&emsp; `RPN`是一个用于生成候选框的网络，使用`RPN`代替`Fast RCNN`的`Selective Search(SS)`，即`Faster RCNN`。RPN其创新点在于其在`feature map`上选择候选框，与网络其他部分共享参数，加快了网络。
&emsp;&emsp;&emsp;&emsp; `Faster RCNN`网络图如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727185901611.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;&emsp;&emsp;  `RPN`网络图如下：

![](https://img-blog.csdnimg.cn/20190727190244311.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;&emsp;&emsp;在得到`feature map`(假定为13 * 13 * 256 )后，经过一个3 * 3的卷积网络(使每一个中心点获得周围点的receptive field),对得到的新的feature map(13 * 13 * 256)，假定每一个中心点存在ｋ（论文中使用了k=9)个`anchor`，对13 * 13上的每一个点将会有256d的维度，将每一个点的256d的这个vector经过两个1 * 1卷积层(`1 * 1 * 2k` ，`1* 1 * 4k`)，一个用于分类(是否为物体2 * k)，一个用于得到anchor内具体的region proposal (即4 * k，只有在分类中是物体时才会进行这个预测)
&emsp;&emsp;&emsp;&emsp;完整的RPN网络如下:
​	
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727191220828.jpg)




# 四、test阶段
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727190032436.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
> 假设该模型为N分类结果，一些细节将会在下文进行介绍
 - 输入一张图片
 - **特征提取**：将整张图片放入卷积网络，得到feature map
 - **候选框提取**，将feature map输入RPN得到，得到候选框，从候选框中使用NMS，并选取前300（超参数）个候选框作为ROI。
 - **ROI Pooling**：根据feature map 已经从RPN得到的以ROI 来进行ROI Pooling 得到固定大小的特征向量
 - **分类与回归**：将得到的特征向量放入fc层，将得到的feature，分别放入bbox回归分类器以及softmax分数分类器，对bbox的位置信息进行修正，并得到分类信息。

<br>

# 五、train阶段
> 根据test阶段，我们可以知道Faster RCNN主要有两个训练任务，为：
> 1、对Fast RCNN的训练
> 2、对RPN的训练
> Faster RCNN是一个two-stage即分两步的训练模式，且对RPN网络的训练以及Fast RCNN网络的训练有独特的顺序。
> 下文将会对其细节进行讲解


## &emsp;&emsp;1、对Fast RCNN的训练
&emsp;&emsp;&emsp;&emsp; 与Fast RCNN网络的训练方法大致相似。
## &emsp;&emsp;2、对RPN的训练
&emsp;&emsp;&emsp;&emsp;　在训练过程中：
&emsp;&emsp;&emsp;&emsp;　对**分类网络的训练**，设定anchor与`ground-truth`的IOU大于0.７的设为正例，小于0.3的设为反例。同时，作者说明那些既不是正例，也不是反例的anchor对网络的训练没有影响。
&emsp;&emsp;&emsp;&emsp;　对**回归网络的训练**，与Fast RCNN对回归网络的训练相似。
&emsp;&emsp;&emsp;&emsp;　同时作者将两个训练的Loss联合起来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727193138257.png)
&emsp;&emsp;&emsp;&emsp;　同时作者发现，参数 `lambda` 对网络的影响不大。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727193444832.png)
&emsp;&emsp;&emsp;&emsp; 训练过程中，每一个mini batch中，作者在一整张图片中随机的选取了256个anchors（正例与反例比值1:1），进行训练。

## &emsp;&emsp;3、整个网络的联合训练
&emsp;&emsp;&emsp;&emsp; 作者在论文中提到，对整个网络的联合训练有三种方式，`Alternating traing`、`Approximate joint training`、`Non-approximate joint training`。
&emsp;&emsp;&emsp;&emsp;  作者采用了第一种训练方法（其他过程可查阅论文)，分为四步
&emsp;&emsp;&emsp;&emsp; 　1、首先训练一次RPN，方法如第二点中所示，且feature map提取网络采用ImageNet-pre-trained的model
&emsp;&emsp;&emsp;&emsp; 　2、然后使用有第一步生成的`proposal`，去训练Fast RCNN，且feature map提取网络也采用ImageNet-pre-trained的model，这两个与训练的网络步是同一个，即暂时不共享网络参数。
&emsp;&emsp;&emsp;&emsp; 　3、使用Fast RCNN的提取feature map的model，微调RPN的网络层。此时，两个网络开始共享参数了。
&emsp;&emsp;&emsp;&emsp; 　4、保持model不变，使用生成的proposal微调Fast RCNN
&emsp;&emsp;&emsp;&emsp;这个过程不断迭代，来不断对网络进行优化。

<br>

# 六、实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727195006752.png)
&emsp;&emsp;&emsp;&emsp;Faster RCNN　在test阶段，198ms 一张图片；Fast RCNN　320ms 一张图片,加快了网络速度.
<br>

# 七、论文细节与思考
## &emsp;&emsp;1、Translation-Invariant Anchors（平移不变性)
>&emsp;&emsp;&emsp;&emsp; Translation-Invariant Anchors　是指当一个物体发生翻转、平移、缩放时，检测网络依旧能很好的检测出来。由于网络中各种各样的anchor以及与anchor相关的回归网络是的Faster RCNN具备Translation-Invariant。
## &emsp;&emsp;2、anchor的尺寸选择
>&emsp;&emsp;&emsp;&emsp; 作者在论文中提到，使用３种scale或者３种aspect rations会使网络具有很好的效果，不过同时使用３种scale和３种aspect rations对网络提升不大，不过作者为了提高网络的灵活程度，同时使用了３种scale和３种aspect rations。
>&emsp;
>![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727200203802.png)
## &emsp;&emsp;3、ROI数量是否越多越好
>&emsp;&emsp;&emsp;&emsp; 作者通过比较发现，RPN网络对ＲＯＩ的数量并不敏感，相反，SS(Selective search）对其比较敏感。这说明NMS并不会损失mAP，并且可能会减少错误。同时作者发现当更改ROI的数量时，其recall基本保持不变，而SS等其他方法变化较大，这解释了为什么ROI数量只有300依旧可以达到很好的效果
> &emsp;     ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727200716865.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201014788.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

## &emsp;&emsp;4、one-stage?two-stage?multi-stage?
>&emsp;&emsp;&emsp;&emsp; 此处引用他人博客
>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201247476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
### &emsp;&emsp; 目标检测的流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201352106.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
### &emsp;&emsp; multi-stage步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201416901.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
### &emsp;&emsp; two-stage步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201443459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

### &emsp;&emsp; one-stage步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190727201456415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
# 八、论文不足

