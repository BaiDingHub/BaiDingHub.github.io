---
title: 目标检测（一）之 R-CNN
date: 2020-04-03 12:01:05
tags:
 - [深度基础知识]
 - [目标检测]
categories: 
 - [深度学习,目标检测]
keyword: "深度学习,目标检测,R-CNN"
description: "目标检测(一)之 R-CNN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(%E4%B8%80)%E4%B9%8B%20R-CNN/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息
## &emsp;&emsp;1.论文题目
&emsp;&emsp;&emsp;&emsp;**Rich feature hierarchies for accurate object detection and semantic segmentation**
## &emsp;&emsp;2.论文时间
&emsp;&emsp;&emsp;&emsp;**2013年**
## &emsp;&emsp;2.论文文献
&emsp;&emsp;&emsp;&emsp; https://arxiv.org/abs/1311.2524


## &emsp;&emsp;3.论文源码
&emsp;&emsp;&emsp;&emsp;  暂无


# 二、论文背景及简介
&emsp;  在过去几年，目标检测常用的算法是机器学习的SIFT以及HOG，其使用机器学习方法融合多种低维图像特征和高维上下文环境的复杂融合系统的方法，结果准确率不高，且运行慢。同时，近几年，深度学习异军突起。RCNN作者便想使用卷积网络提取图像特征，用于目标检测。最终R-CNN在pascal VOC 2012数据集上取得了mAP 53.3%的成绩。

<br>

# 三、知识储备
## &emsp;&emsp; 1、selective search
&emsp;&emsp;&emsp;&emsp;  `selective search` 是一种从一张图片中选择region proposal的方法。在之前的机器学习算法中，大多采用回归或者slide window的方法进行候选框的选择，十分耗时，该篇论文使用selective search 大大节省了时间。
&emsp;&emsp;&emsp;&emsp; `selective search` 根据图片的颜色直方图等大量的信息来对图片进行提取，其大体步骤如下：
- step1:计算区域集R里每个相邻区域的相似度S={s1,s2,…} 
- step2:找出相似度最高的两个区域，将其合并为新集，添加进R 
- step3:从S中移除所有与step2中有关的子集 
- step4:计算新集与所有子集的相似度 
- step5:跳至step2，直至S为空

&emsp;&emsp;&emsp;&emsp; 可以看[selective search的论文](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)进行更深一步了解

## &emsp;&emsp; 2、IOU
&emsp;&emsp;&emsp;&emsp;  IOU是计算矩形框A、B的重合度的公式：IOU=(A∩B)/(A∪B)，重合度越大，说明二者越相近。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716234801901.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

## &emsp;&emsp; 3、mAP(mean Average Precision)
&emsp;&emsp;&emsp;&emsp;所有类AP(Average Precision)值的平均值

## &emsp;&emsp;  4、非极大值抑制(NMS)
&emsp;&emsp;&emsp;&emsp; 当在图像中预测多个`bbox`时，由于预测的结果间可能存在高冗余（即同一个目标可能被预测多个矩形框），因此可以过滤掉一些彼此间高重合度的结果
&emsp;&emsp;&emsp;&emsp; 具体操作就是根据各个bbox的score降序排序，剔除与高score bbox有较高重合度的低score bbox，那么重合度的度量指标就是IoU；

## &emsp;&emsp;  5、Bounding-box regression
&emsp;&emsp;&emsp;&emsp;  经过`NMS`获得的候选框往往与`ground-truth box`有一定的差距，因此，我们需要通过Bounding-box regression来对得到的候选框进行一定程度的修正。
&emsp;&emsp;&emsp;&emsp;   可以设`region proposal` 为 `P = (Px,Py,Pw,Ph)` , 设 `ground-truth box` 为 `G = (Gx,Gy,Gw,Gh)`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716235740614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716235746730.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)


# 四、test阶段
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190716215853278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
> 假设该模型为N分类结果，一些细节将会在下文进行介绍
 - 输入一张图片
 - **候选框提取**：使用`selective search`方法，获取该图片的候选框(`region proposal`),一般有2000多个，这里假设为2000
 - **特征提取**：将得到的候选区域，`resize`后，送入卷积网络，在最后得到特征图(`feature map`)
 - **分类**：根据`feature map`使用线性SVM(每一个分类有一个SVM分类器)，对得到的候选框进行打分 ，得到打分的矩阵(2000 * 20)
 - **候选框选择**：使用非极大值抑制(`NMS`)，去除重复率高的一些候选框
 - **回归**：对最后的候选框进行回归修正(`Bounding-box regression`)，得到最后的`Bounding box`（`b-box`) 以及分类结果

<br>

# 五、train阶段
> 根据test阶段，我们可以知道在训练阶段，主要有三个任务，1、卷积网络的确定以及fine tuning  &emsp;&emsp;2、SVM的训练 &emsp;&emsp; 3、Bounding-box regression的训练  
> 下文将会对其原因以及细节进行讲解

## &emsp;&emsp;1、卷积网络的确定以及fine tuning
&emsp;&emsp;&emsp;&emsp; 检测问题中由于带标签的样本数据量比较少，难以进行大规模训练，因此在RCNN中，卷积网络使用在imageNet数据集上预训练好的AlexNet。当应用到自己的数据集上时，要进行适当的fine tuning。
&emsp;&emsp;&emsp;&emsp;在fine tuning 过程中，作者将`region proposal`与`ground-truth box` 的`IOU`大于`0.5`的设为正例(`positive`)，剩下的作为反例(`negative`)。取代AlexNet的`1000 way`分类器，增加全连接层`N+1 way`（还有背景这个标签） 进行分类。一个`batch`有128个样本，其中32个正例，96个反例，因为`IOU`大于0.5的太少了。设置`learning rate = 0.001` 使用`SGD` 进行fine tuning


## &emsp;&emsp;2、SVM的训练
&emsp;&emsp;&emsp;&emsp; 在得到卷积网络后，下一步应该就要训练好分类器。在这里，作者使用了SVM对每一个分类进行二分类，而没有用softmax，具体细节可看下文细节说明。因此，N分类任务，就要训练N个SVM。
&emsp;&emsp;&emsp;&emsp; 需要注意的是，在训练SVM时，其训练样本采用了不同的标注方式，在这里，作者只将`ground-truth box`作为`positive` , 将`IOU` 小于 `0.3` 的设为`negative`,0.3是作者调参调出来的。梯度下降来对SVM进行训练即可。
## &emsp;&emsp;3、Bounding-box regression的训练  
>需要为每一个类别都训练一个regression
>Bounding-box regression的完整训练过程将在本文的细节部分进行介绍

&emsp;&emsp;&emsp;&emsp; 经过`NMS`获得的候选框往往与`ground-truth box`有一定的差距，因此，在论文中便使用Bounding-box regression的方式来对得到的候选框进行修正。
&emsp;&emsp;&emsp;&emsp; 将`NMS` 获得的每个类别的`region proposal（x、y、w、h)`，`pool5`输出的特征图（6 * 6 * 256维）作为输入，根据`ground-truth box`进行回归的训练。
<br>
# 六、实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717002041623.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717002047838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
<br>

# 七、论文细节
## &emsp;&emsp;1、为什么SVM训练与fine tuning的样本采样方式不同？
>&emsp;&emsp;&emsp;&emsp; 其实，作者实现进行的SVM训练，之后再想到的fine tuning。由于SVM有较少的样本依旧训练的很好，所有作者给了SVM比较严格的样本点，数量少，但仍然可以训练的比较好。但是在卷积网络上，样本点太少，很难训练，所有作者利用了IOU 0.5~1之间的样本及进行fine turing

## &emsp;&emsp;2、在Bounding-box regression 为何使用pool5输出的特征图？
>&emsp;&emsp;&emsp;&emsp;在论文中，作者对`AlexNet`的`pool5`、`fc6`、`fc7`得到的特征图进行了可视化的分析，作者发现，`pool5`的`feature map`与类别更加无关，所以可以把`pool5`之前的部分充当特征提取器，而`fc6 fc7` 更适用于根据特征图进行分类，如果想要深入了解，请看论文。

## &emsp;&emsp;3、在得到region proposal后，送入卷积网络时要进行缩放，缩放方式是怎样的？
&emsp;&emsp;&emsp;&emsp;在附录A中讨论了很多的预处理方法，

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717000757345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;&emsp;&emsp;A. 原图
&emsp;&emsp;&emsp;&emsp;B. 等比例缩放，空缺部分用原图填充
&emsp;&emsp;&emsp;&emsp;C. 等比例缩放，空缺部分填充bounding box均值
&emsp;&emsp;&emsp;&emsp;D. 不等比例缩放到224x224

&emsp;&emsp;&emsp;&emsp;实验结果表明B的效果最好，但实际上还有很多的预处理方法可以用，比如空缺部分用区域重复。


<br>

# 八、论文不足
&emsp;&emsp;&emsp;&emsp;由于R-CNN流程众多，包括region proposal的选取，训练卷积神经网络，训练SVM和训练 regression，而且，对得到的每一个`region proposal`都要进行卷积，使得其**训练时间很长**（84小时），而且中间结果都要保存，**占用磁盘大**，而且**测试一张图片的时间也较长**`(13s/image on a GPU or 53s/image on a CPU)`。
