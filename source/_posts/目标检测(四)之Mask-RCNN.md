---
title: 目标检测（四）之Mask-RCNN
date: 2020-04-03 12:04:05
tags:
 - [深度基础知识]
 - [目标检测]
categories: 
 - [深度学习,目标检测]
keyword: "深度学习,目标检测,Mask-RCNN"
description: "目标检测（四）之Mask-RCNN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(%E5%9B%9B)%E4%B9%8BMask-RCNN/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息
## &emsp;&emsp;1.论文题目
&emsp;&emsp;&emsp;&emsp;**Mask-RCNN**
## &emsp;&emsp;2.论文时间
&emsp;&emsp;&emsp;&emsp;**2017年**

## &emsp;&emsp;3.论文文献
&emsp;&emsp;&emsp;&emsp; [论文文献](https://arxiv.org/abs/1703.06870)
## &emsp;&emsp;4.论文源码
&emsp;&emsp;&emsp;&emsp;  [pytroch](https://github.com/facebookresearch/maskrcnn-benchmark)


# 二、论文背景及简介

&emsp;&emsp;&emsp;&emsp;在目标检测领域，RCNN、Fast RCNN，Faster RCNN做出了突出的贡献，Mask RCNN是大神何凯明2017年对Faster RCNN的改进，使用ROI Align修正了Faster RCNN中ROI pooling不精确的问题，同时增加了mask的输出，使得Mask RCNN可以应用于语义分割、人姿势检测等多个场景，同时该模型在2017年获得了多个领域的第一名。

# 三、知识储备
## 1、ROI Align
&emsp;&emsp;&emsp;&emsp; 转载于：https://blog.csdn.net/Bruce_0712/article/details/80287385
### **ROI pooling的局限性**
&emsp;&emsp;&emsp;&emsp; `ROI Align`实际上是对`ROI pooling`的精度的改进，下面我们先来看一下`ROI pooling`存在哪些问题。
&emsp;&emsp;&emsp;&emsp; `ROI Pooling` 的作用是根据预选框的位置坐标在特征图中将相应区域池化为固定尺寸的特征图，以便进行后续的分类和包围框回归操作。由于预选框的位置通常是由模型回归得到的，一般来讲是浮点数，而池化后的特征图要求尺寸固定。故`ROI Pooling`这一操作存在两次量化（`Quantization`）的过程。
   - 将候选框边界量化为整数点坐标值。
   - 将量化后的边界区域平均分割成 k x k 个单元(bin),对每一个单元的边界进行量化。

&emsp;&emsp;&emsp;&emsp;  事实上，经过上述两次量化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。在论文里，作者把它总结为“不匹配问题（misalignment）。
&emsp;&emsp;&emsp;&emsp; 下面我们用直观的例子具体分析一下上述区域不匹配问题。
&emsp;&emsp;&emsp;&emsp;  如 图1 所示，这是一个`Faster-RCNN`检测框架。输入一张`800*800`的图片，图片上有一个`665*665`的包围框(框着一只狗)。图片经过主干网络提取特征后，特征图缩放步长（stride）为`32`。因此，图像和包围框的边长都是输入时的`1/32`。`800`正好可以被`32`整除变为`25`。但`665`除以`32`以后得到`20.78`，带有小数，于是`ROI Pooling` 直接将它量化成`20`。接下来需要把框内的特征池化`7*7`的大小，因此将上述包围框平均分割成`7*7`个矩形区域。显然，每个矩形区域的边长为`2.86`，又含有小数。于是`ROI Pooling` 再次把它量化到`2`。经过这两次量化，候选区域已经出现了较明显的偏差（如图中绿色部分所示）。更重要的是，该层特征图上0.1个像素的偏差，缩放到原图就是3.2个像素。那么0.8的偏差，在原图上就是接近30个像素点的差别，这一差别不容小觑。
<center> 图1</center >

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804131434345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

### &emsp;&emsp; **ROI Align的思想**
&emsp;&emsp;&emsp;&emsp; 为了解决ROI Pooling的上述缺点，作者提出了ROI Align这一改进的方法(如图2)。ROI Align的思路很简单：取消量化操作，使用`双线性内插`的方法获得坐标为浮点数的像素点上的图像数值,从而将整个特征聚集过程转化为一个连续的操作，。值得注意的是，在具体的算法操作上，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后将这些坐标点进行池化，而是重新设计了一套比较优雅的流程，如 图3 所示：

- 遍历每一个候选区域，保持浮点数边界不做量化。
- 将候选区域分割成k x k个单元，每个单元的边界也不做量化。
- 在每个单元中计算固定四个坐标位置，用双线性内插的方法计算出这四个位置的值，然后进行最大池化操作。

&emsp;&emsp;&emsp;&emsp; 这里对上述步骤的第三点作一些说明：这个固定位置是指在每一个矩形单元（bin）中按照固定规则确定的位置。比如，如果采样点数是1，那么就是这个单元的中心点。如果采样点数是4，那么就是把这个单元平均分割成四个小方块以后它们分别的中心点。显然这些采样点的坐标通常是浮点数，所以需要使用插值的方法得到它的像素值。在相关实验中，作者发现将采样点设为4会获得最佳性能，甚至直接设为1在性能上也相差无几。事实上，ROI Align 在遍历取样点的数量上没有ROIPooling那么多，但却可以获得更好的性能，这主要归功于解决了misalignment的问题。值得一提的是，我在实验时发现，ROI Align在VOC2007数据集上的提升效果并不如在COCO上明显。经过分析，造成这种区别的原因是COCO上小目标的数量更多，而小目标受misalignment问题的影响更大（比如，同样是0.5个像素点的偏差，对于较大的目标而言显得微不足道，但是对于小目标，误差的影响就要高很多）。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804131856588.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
<br>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804131859746.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
### &emsp;&emsp; **ROI Align的反向传播**
&emsp;&emsp;&emsp;&emsp; 常规的ROI Pooling的反向传播公式如下：

$$
\frac {\partial L}{\partial x_i} = \sum_{r}\sum_{j}[i =i*(r,j)]\frac {\partial L}{\partial y_rj} 
$$


&emsp;&emsp;&emsp;&emsp; 这里，`x_i`代表池化前特征图上的像素点；`y_rj`代表池化后的第r个候选区域的第`j`个点；`i*(r,j)`代表点yrj像素值的来源（最大池化的时候选出的最大像素值所在点的坐标）。由上式可以看出，只有当池化后某一个点的像素值在池化过程中采用了当前点Xi的像素值（即满足`i=i*(r，j)）`，才在`x_i`处回传梯度。

&emsp;&emsp;&emsp;&emsp; 类比于ROIPooling，ROIAlign的反向传播需要作出稍许修改：首先，在ROIAlign中，`i*（r,j）`是一个浮点数的坐标位置(前向传播时计算出来的采样点)，在池化前的特征图中，每一个与 `i*(r,j)` 横纵坐标均小于1的点都应该接受与此对应的点`y_rj`回传的梯度，故ROI Align 的反向传播公式如下: 
　　 
$$
\frac {\partial L}{\partial x_i} = \sum_{r}\sum_{j}[d(i =i*(r,j))<1](1-\Delta h)(1-\Delta w)\frac {\partial L}{\partial y_rj} 
$$

&emsp;&emsp;&emsp;&emsp; 上式中，`d(.)`表示两点之间的距离，`Δh`和`Δw`表示`x_i` 与 `i*(r,j)` 横纵坐标的差值，这里作为双线性内插的系数乘在原始的梯度上。



# 四、test阶段
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804133606329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804133618852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

> 假设该模型为K分类结果，一些细节将会在下文进行介绍
 - 输入一张图片
 - **特征提取**：将整张图片放入卷积网络，得到feature map
 - **候选框提取**，将feature map输入RPN得到，得到候选框，从候选框中使用NMS，并选取前300（超参数）个候选框作为ROI。
 - **ROI Algin**：根据feature map 已经从RPN得到的以ROI 来进行ROI Align得到固定大小的特征向量
 - **分类与回归**：将得到的特征向量放入fc层，将得到的feature，分别放入bbox回归分类器以及softmax分数分类器，对bbox的位置信息进行修正，并得到分类信息。
 - **mask预测** 将得到的特征向量 放入 卷积层，预测mask

<br>

# 五、train阶段
> 根据test阶段，我们可以知道Faster RCNN主要有两个训练任务，为：
> 1、对Faster RCNN的训练
> 2、对mask预测网络的训练。
> 下文将会对其细节进行讲解


## 1、对Faster RCNN的训练
&emsp;&emsp;&emsp;&emsp; 与Faster RCNN网络的训练方法大致相似。
## 2、对mask预测网络的训练
&emsp;&emsp;&emsp;&emsp;  mask预测网络的输出结果是K*m*m维向量，其中K为类别数量，m*m是mask的最终输出长宽。计算每个像素的sigmoid结果，最终Mask 损失就是二维交叉熵损失的平均值（average binary cross-entropy loss）。
&emsp;&emsp;&emsp;&emsp; 值得注意的是，在mask预测网络中，输出K 个 mask，根据预测的class k来选取第k个mask来作为结果，这减少了类之间的竞争。
&emsp;&emsp;&emsp;&emsp; 确定了L_mask后，作者将L_mask,L_cls,L_box相加来作为整个网络的loss一起训练。

## 3、训练参数
- 每个GPU同时训练两张图片（作者用了8GPU，所以batch size是16），输入图片尺寸为800*800。
- 训练时，每张图片的RoI数量为64/512（根据基础网络不同而改变）；测试时每张图片RoI数量为300/1000。
- 正反例比例为1:3。
- anchors使用 5 scales 和 3 aspect ratios。
- weight decay为0.0001。
- 学习率：0.02，到120k iteration后为除以10。
# 六、实验结果
`Mask RCNN` 达到 `state-of-the-art` 的效果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804134722674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804134727582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
<br>

# 七、论文细节与思考
## 1、ROI Align对精度要求高的训练任务更有用
>&emsp;&emsp;&emsp;&emsp; ROI pooling虽然损失了精度，但是对目标检测这种精度需求不高的任务中，仍然能获得很好的效果，但很难用在语义分割或者人姿势检测这些任务中，其要求精度很高，所以ROI Align效果更好

## 2、增添了mask 预测网络对目标检测是否有用
>&emsp;&emsp;&emsp;&emsp; 作者做了ablation experiment，发现去掉mask 预测网络，会对模型的结果有影响。
>![在这里插入图片描述](https://img-blog.csdnimg.cn/20190804135240949.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
## 3、Mask RCNN如何应用在姿势检测中
>&emsp;&emsp;&emsp;&emsp; 假设姿势检测任务要求输出K个关键点，则Mask的将会输出K*m*m的特征图，每一个m*m的特征图用于对一个特定关键点的检测。且该特征图是一个二值矩阵。

# 八、论文优缺点
## 优点
- Mask RCNN将语义分割与目标检测通过输出mask的方式进行结合，加强了目标检测的准确度。
- Mask RCNN将ROI pooling优化为ROI Align，解决了ROI pooling的量化问题，提高了定位小物体以及mask预测的精度

## 缺点
