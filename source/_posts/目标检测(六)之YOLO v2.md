---
title: 目标检测（六）之YOLO v2
date: 2020-04-03 12:06:05
tags:
 - [深度基础知识]
 - [目标检测]
categories: 
 - [深度学习,目标检测]
keyword: "深度学习,目标检测,YOLO v2"
description: "目标检测（六）之YOLO v2"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(%E5%85%AD)%E4%B9%8BYOLO%20v2/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息
## &emsp;&emsp;1.论文题目
&emsp;&emsp;&emsp;&emsp;**YOLO9000: Better, Faster, Stronger**
## &emsp;&emsp;2.论文时间
&emsp;&emsp;&emsp;&emsp;**2016年**

## &emsp;&emsp;3.论文文献
&emsp;&emsp;&emsp;&emsp; [论文文献](https://arxiv.org/abs/1612.08242)
## &emsp;&emsp;4.论文源码
&emsp;&emsp;&emsp;&emsp;  pytroch


# 二、论文背景及简介

&emsp;&emsp;&emsp;&emsp;YOLO v2是对YOLO v1版本的改进，使得YOLO Better, Faster, Stronger
&emsp;&emsp;&emsp;&emsp;其主要有两个大方面的改进：

&emsp;&emsp;&emsp;&emsp; **第一，作者使用了一系列的方法对原来的YOLO多目标检测框架进行了改进，在保持原有速度的优势之下，精度上得以提升**。VOC 2007数据集测试，67FPS下mAP达到76.8%，40FPS下mAP达到78.6%，基本上可以与Faster R-CNN和SSD一战。这一部分是本文主要关心的地方。

&emsp;&emsp;&emsp;&emsp; **第二，作者提出了一种目标分类与检测的联合训练方法，通过这种方法，YOLO9000可以同时在COCO和ImageNet数据集中进行训练，训练后的模型可以实现多达9000种物体的实时检测**。


# 三、YOLO v2的一系列改进
## 1、总览
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808151450656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
## 2、Better
## (1) Batch Normalization
&emsp;&emsp;&emsp;&emsp;CNN在训练过程中网络每层输入的分布一直在改变, 会使训练过程难度加大，但可以通过normalize每层的输入解决这个问题。新的YOLO网络在每一个卷积层后添加batch normalization，通过这一方法，mAP获得了2%的提升。batch normalization 也有助于规范化模型，可以在舍弃dropout优化后依然不会过拟合。
## (2) High Resolution Classifier
&emsp;&emsp;&emsp;&emsp; 目前的目标检测方法中，基本上都会使用ImageNet预训练过的模型（classifier）来提取特征，如果用的是AlexNet网络，那么输入图片会被resize到不足256 * 256，导致分辨率不够高，给检测带来困难。为此，新的YOLO网络把分辨率直接提升到了448 * 448，这也意味之原有的网络模型必须进行某种调整以适应新的分辨率输入。

&emsp;&emsp;&emsp;&emsp; 对于YOLOv2，作者首先对自己在ImageNet上训练好的分类网络（自定义的darknet）进行了fine tune，分辨率改成448 * 448，在ImageNet数据集上训练10轮（10 epochs），训练后的网络就可以适应高分辨率的输入了。然后，作者对检测网络部分（也就是后半部分）也进行fine tune。这样通过提升输入的分辨率，mAP获得了4%的提升。
## (3) Convolutional With Anchor Boxes
&emsp;&emsp;&emsp;&emsp;  之前的YOLO利用全连接层的数据完成边框的预测，导致丢失较多的空间信息，定位不准。作者在这一版本中借鉴了Faster R-CNN中的anchor思想。
&emsp;&emsp;&emsp;&emsp; 作者去掉了后面的一个池化层以确保输出的卷积特征图有更高的分辨率。然后，通过缩减网络，让图片输入分辨率为416 * 416，这一步的目的是为了让后面产生的卷积特征图宽高都为奇数，这样就可以产生一个center cell。作者观察到，大物体通常占据了图像的中间位置， 就可以只用中心的一个cell来预测这些物体的位置，否则就要用中间的4个cell来进行预测，这个技巧可稍稍提升效率。最后，YOLOv2使用了卷积层降采样（factor为32），使得输入卷积网络的416 * 416图片最终得到13 * 13的卷积特征图（416/32=13）。加入了anchor boxes后，使得结果是召回率上升，准确率小幅度下降。具体数据为：没有anchor boxes，模型recall为81%，mAP为69.5%；加入anchor boxes，模型recall为88%，mAP为69.2%。
## (4) Dimension Clusters（维度聚类）
&emsp;&emsp;&emsp;&emsp;  作者在使用anchor的时候遇到了两个问题，第一个是anchor boxes的宽高维度往往是精选的先验框（hand-picked priors），虽说在训练过程中网络也会学习调整boxes的宽高维度，最终得到准确的bounding boxes。但是，如果一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就更容易学到准确的预测位置。

&emsp;&emsp;&emsp;&emsp;  和以前的精选boxes维度不同，作者使用了**K-means聚类方法**训练bounding boxes，可以自动找到更好的boxes宽高维度。传统的K-means聚类方法使用的是欧氏距离函数，也就意味着较大的boxes会比较小的boxes产生更多的error，聚类结果可能会偏离。为此，作者采用的评判标准是IOU得分（也就是boxes之间的交集除以并集），这样的话，error就和box的尺度无关了，最终的距离函数为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808152447868.png)
&emsp;&emsp;&emsp;&emsp; 作者在VOC和COCO数据集上均做了聚类，其结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808152710710.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;&emsp;&emsp; 作者在平衡平均IOU和网络计算复杂度的情况下选择了k = 5，同时作者发现聚类的结果显示，在检测框中细长的框多，扁宽的少。同时，作者将其与传统选择anchor的方法进行了比较，结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808152915177.png)
&emsp;&emsp;&emsp;&emsp; 可以看到在聚类情况下，5个anchor已经可以与没有聚类的9个anchor相比了。 

## (5) Direct loaction prediction（最后的回归修正过程）
&emsp;&emsp;&emsp;&emsp; **在基于region proposal的目标检测算法中，是通过预测`tx`和`ty`来得到`(x,y)`值，也就是预测的是`offsets`。**
&emsp;&emsp;&emsp;&emsp; 论文这里公式是错的，应该是“+”号。依据是下文中的例子，以及Faster R-CNN中的公式。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808163432859.jpg)

&emsp;&emsp;&emsp;&emsp; **这个公式是无约束的，预测的边界框很容易向任何方向偏移。**
&emsp;&emsp;&emsp;&emsp; &emsp;&emsp; 当`tx=1`时，box将向右偏移一个anchor box的宽度；
&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;   当`tx=-1`时，box将向左偏移一个anchor box的宽度；
&emsp;&emsp;&emsp;&emsp; **因此，每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的offsets。**

&emsp;&emsp;&emsp;&emsp;YOLOv2中没有采用这种预测方式，而是沿用了YOLOv1的方法，就是预测边界框中心点相对于对应`cell`**左上角位置的相对偏移值。**
&emsp;&emsp;&emsp;&emsp;网络在最后一个卷积层输出`13*13`的`feature map`，有`13*13`个cell，每个cell有5个anchor box来预测5个bounding box，每个bounding box预测得到5个值。
&emsp;&emsp;&emsp;&emsp;分别为：`tx`、`ty`、`tw`、`th`和`to`（类似YOLOv1的confidence）
&emsp;&emsp;&emsp;&emsp;为了将bounding box的中心点约束在当前cell中，使用`sigmoid函数`将`tx`、`ty`归一化处理，将值约束在`0~1`，这使得模型训练更稳定。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808163903124.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808155021350.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

## &emsp;&emsp;&emsp; (5) Fine-Grained Features（细粒度特征）
&emsp;&emsp;&emsp;&emsp; 作者在网络中添加了个`passthrough layer`，这个layer也就是把高低两种分辨率的特征图做了一次连接，连接方式是叠加特征到不同的通道而不是空间位置，类似于Resnet中的identity mappings。这个方法把26 * 26 * 512的特征图连接到了13 * 13 * 2048的特征图，这个特征图与原来的特征相连接。YOLO的检测器使用的就是经过扩张的特征图，它可以拥有更好的细粒度特征，使得模型的性能获得了1%的提升。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808155313984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

## (6) Multi-Scale Training 
&emsp;&emsp;&emsp;&emsp; YOLO v2的网络结构中由于没有了全连接层，全部由卷积和池化构成，因此输入的图片大小可以是任意的。作者为了是网络适应不同大小的输入，进行了Multi-Scale Training。作者是将输入图片，每10batches，就更改一下输入图片的尺寸（32的倍数{320,352,.....,608}）
&emsp;&emsp;&emsp;&emsp; 这种机制使得网络可以更好地预测不同尺寸的图片，意味着同一个网络可以进行不同分辨率的检测任务，在小尺寸图片上YOLOv2运行更快，在速度和精度上达到了平衡。

&emsp;&emsp;&emsp;&emsp; 在小尺寸图片检测中，YOLOv2成绩很好，输入为228 * 228的时候，帧率达到90FPS，mAP几乎和Faster R-CNN的水准相同。使得其在低性能GPU、高帧率视频、多路视频场景中更加适用。

&emsp;&emsp;&emsp;&emsp; 在大尺寸图片检测中，YOLOv2达到了先进水平，VOC2007 上mAP为78.6%，仍然高于平均水准，下图是YOLOv2和其他网络的成绩对比：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808155747356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

## 3、Faster
>作者为了改善检测速度，也作了一些相关工作。
## (1) Darknet-19 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808160239576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
##  (2) Training for classification
&emsp;&emsp;&emsp;&emsp;  作者使用Darknet-19在标准1000类的ImageNet上训练了160次，用的随机梯度下降法，starting learning rate 为0.1，polynomial rate decay 为4，weight decay为0.0005 ，momentum 为0.9。训练的时候仍然使用了很多常见的数据扩充方法（data augmentation），包括random crops, rotations, and hue, saturation, and exposure shifts。 （这些训练参数是基于darknet框架，和caffe不尽相同）

&emsp;&emsp;&emsp;&emsp;  初始的224 * 224训练后，作者把分辨率上调到了448 * 448，然后又训练了10次，学习率调整到了0.001。高分辨率下训练的分类网络在top-1准确率76.5%，top-5准确率93.3%。
## (3) Training for detection
&emsp;&emsp;&emsp;&emsp;  分类网络训练完后，就该训练检测网络了，作者去掉了原网络最后一个卷积层，转而增加了三个3 * 3 * 1024的卷积层，并且在每一个上述卷积层后面跟一个1 * 1的卷积层，输出维度是检测所需的数量。对于VOC数据集，预测5种boxes大小，每个box包含5个坐标值和20个类别，所以总共是5 * （5+20）= 125个输出维度。同时也添加了转移层（passthrough layer ），从最后那个3 * 3 * 512的卷积层连到倒数第二层，使模型有了细粒度特征。

作者的检测模型以0.001的初始学习率训练了160次，在60次和90次的时候，学习&emsp;&emsp;&emsp;&emsp;  率减为原来的十分之一。其他的方面，weight decay为0.0005，momentum为0.9，依然使用了类似于Faster-RCNN和SSD的数据扩充（data augmentation）策略。

## 4、Stronger
> YOLO 9000的由来，可以同时检测9000中类别。（ps：YOLO v1是20分类）
> 在这里我们简介一下思想，具体内容请查看论文（我理解的也不是很好）

&emsp;&emsp;&emsp;&emsp; 人为对图像的目标进行标注的代价是巨大的，因此没有那么大的数据集来支撑那么多类别的检测，那么作者是如何解决这个问题的呢？
&emsp;&emsp;&emsp;&emsp; 原来，作者将目光瞄向了Image Net。作者通过一种神奇的训练方法将分类与目标检测联合起来训练，这使得该网络具备了更多分类的能力，当然，在这些数据上，精度稍微小一点。那么作者是如何做的呢？
&emsp;&emsp;&emsp;&emsp; 首先介绍一下，作者是怎样进行训练的。在网络处理过程中，如果输入的是图片分类的图片，那么网络只反向传播分类的那一块loss，如果输入的是目标检测的图片，那么网络将进行正常的反向传播，来对目标检测和一块进行优化。通过这种方法，作者就使得网络具备了更多分类的能力。因此在分类输出那一块是经过softmax直接输出1000多种的分类结果
&emsp;&emsp;&emsp;&emsp; 但这样处理的时候，又出现了新的问题。假设ImageNet收录了哈士奇和藏獒的图片，但没有牧羊犬的图片，那么当放入牧羊犬的图片是，网络该如何做呢？
&emsp;&emsp;&emsp;&emsp; 我们知道哈士奇和藏獒都属于狗，那能不能在softmax输出的时候，输出多个标签，比如输出哈士奇和狗，这样在遇到牧羊犬的时候，只输出狗的标签就可以了。因此作者对ImageNet数据集做出了修改。
&emsp;&emsp;&emsp;&emsp;ImageNet数据集的标签是来自于WordNet的，一个语言的有向图，这里面包含了各种分类之间的关系。为了简化，作者将ImageNet的标签构造成了一颗树，这棵树从上到下就是隶属关系。之后作者也将COCO的数据集加入到了这颗树里面。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808162007736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;&emsp;&emsp; 假如ImageNet 有1000个分类，经过构造树后，得到了369个根节点，那么softmax就输出1369个分类结果，在给出某个类别的预测概率时，需要找到其所在的位置，遍历这个path，然后计算path上各个节点的概率之积。根据最大的置信度来对该图片进行分类，这也就解决了上述问题。
&emsp;&emsp;&emsp;&emsp;  最后作者联合了多个数据集，将种类数提高到了9418。也就是说YOLO9000可以同时对9000多种目标进行检测。这就是论文题目的由来



# 四、实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808162635121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
