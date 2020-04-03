---
title: 对抗样本（四）BIM&ILLC
date: 2020-04-03 13:04:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,BIM&ILLC"
description: "对抗样本（四）BIM&ILLC"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%9B%9B%EF%BC%89BIM&ILLC/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **Adversarial examples in the physical world**

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2016年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp;[https://arxiv.org/abs/1607.02533](https://arxiv.org/abs/1607.02533)



# 二、论文背景及简介

 &emsp; &emsp; 目前大多数的攻击对象都是threat model，攻击者可以直接将数据送到分类器中。但在现实世界，可不都是这种情况，比如在很多情况下，人们只能依靠一些设备例如照相机、传感器来传送数据（生成对抗样本后，再由照相机拍照或传感器感知）。这篇文章展示了，在这些现实世界的传感器面前，机器学习系统也是易受对抗样本攻击的。

<br>

# 三、论文内容总结

- 探讨了为在物理世界中运行的机器学习系统创建对抗样本的可能性
- 使用了迭代的方法来生成对抗样本，有**BIM**以及**ILLC**：

$$
X_0^{adv} = X,\ \ \ \ X_{N+1}^{adv} = Clip_{X,\epsilon}\{X_{N}^{adv}+\alpha sign(\nabla_{X}J(X_{N}^{adv},y_{true})\}
$$

<br>

$$
X_0^{adv} = X,\ \ \ \ X_{N+1}^{adv} = Clip_{X,\epsilon}\{X_{N}^{adv}-\alpha sign(\nabla_{X}J(X_{N}^{adv},y_{LL}))\}
$$

- 引入了**破坏率**来表示现实世界中的变换对对抗样本的影响
- ILLC生成的扰动相对于FGSM来说，更小。也更容易被现实世界所破坏

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

<br>

# 四、论文主要内容

## 1、Introduction

 &emsp; &emsp; 在本节，作者主要介绍了现实世界的攻击是怎样的情况。

 &emsp; &emsp; 之前的工作做的都是电脑内部的对抗攻击，比如：逃避恶意软件检测等。而并没有考虑过现实世界的应用，比如：通过照相机和传感器感知世界的机器人，视频监控系统，手机应用等。在这样的情境中，攻击者不能够依赖输入数据的每像素细粒度修改能力。这就产生了一个问题，也是本篇论文主要讨论的问题，是否能够在这样的情况下（应用于现实世界，使用各种传感器来感知数据，而不是通过数学表示），得到对抗样本？

<br>

## 2、Methods Of Generating Adversarial Images

 &emsp; &emsp; 在本节，主要介绍了一些生成对抗样本的不同的方法。**BIM（Basic Iterative Method）** 以及**ILLC（Iteratice Least-Likely Class Method）** 便在本节介绍。

 &emsp; &emsp; 在此，解释一下，本节要用到的一个特殊的符号：
$$
Clip_{X,\epsilon}\{X'\}(x,y,z) = min\{255,X(x,y,z)+\epsilon,max\{0,X(x,y,z)-\epsilon,X'(x,y,z)\}\}
$$
 &emsp; &emsp; 在该公式中，$\ x,y,z$ 表示X处于三维空间中，也即图片的宽度、高度、通道数。该公式的意思是限制生成的对抗样本在X的$\ \epsilon$ 邻域内。

 &emsp; &emsp; 在这节也介绍了FGSM，但本文重点不在此，在这里不多介绍。

 &emsp; &emsp; **BIM是FGSM的拓展，作者进行了多次小步的迭代，并且在每一步之后都修剪得到的结果的像素值**，来确保得到的结果在原始图像的$\ \epsilon$ 邻域内，公式如下：
$$
X_0^{adv} = X,\ \ \ \ X_{N+1}^{adv} = Clip_{X,\epsilon}\{X_{N}^{adv}+\alpha sign(\nabla_{X}J(X_{N}^{adv},y_{true})\}
$$
 &emsp; &emsp; 在实验中，作者使用$\ \alpha = 1$ ，这意味着，在每一步我们改变每一个像素1点。作者选择迭代次数为$\ min(\epsilon+4,1.25\epsilon)$ 。

 &emsp; &emsp; **ILLC是BIM的拓展，ILLC将攻击拓展到了目标攻击**，该迭代方法试图让对抗样本被误分类成一个特定的类，作者选择与原图像最不像的类作为目标类，即：
$$
y_{LL} = argmin_y \{p(y|X)\}
$$
 &emsp; &emsp; 为了让对抗样本误分类成$\ y_{LL}$ ，我们就需要最大化$\ log \ p(y_{LL}|X)$ 。所以我们就需要在$\ sign\{\nabla_X log\ p(y_{LL}|X)\}$的方向上进行迭代，对于使用交叉熵作为损失函数的网络，其表达形式就是$\ sign\{-\nabla_X J(X,y_{LL})\}$ 。因此ILLC的表达形式为：
$$
X_0^{adv} = X,\ \ \ \ X_{N+1}^{adv} = Clip_{X,\epsilon}\{X_{N}^{adv}-\alpha sign(\nabla_{X}J(X_{N}^{adv},y_{LL}))\}
$$
<br>

 &emsp; &emsp; 作者就三个方法进行了实验比较，实验方法如下：

 &emsp; &emsp; 整个实验是在ImageNet的验证集(50000张图片)上进行的，使用了一个预训练的Inception v3的分类器。对每一个验证集上的图片，我们使用不同的方法和$\ \epsilon$ 生成对抗样本，并且记录在全部的50000张图片上的分类正确率。之后，我们计算了在干净数据集上的正确率。结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311192856426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)


<br>

## 3、Photos Of Adversarial Examples

### 1、对抗样本的破坏率(Destruction rate)

 &emsp; &emsp; 为了研究任意变换对对抗样本的影响，作者引入了破坏率的概念。即：在变换后，对抗样本不再被误分类的比例。定义如下：
$$
d = \frac{\sum_{k=1}^nC(X^k,y_{true}^k)\overline{C(X_{adv}^k,y_{true}^k)}C(T(X_{adv}^k),y_{true}^k)}{\sum_{k=1}^nC(X^k,y_{true}^k)\overline{C(X_{adv}^k,y_{true}^k)}}
$$
 &emsp; &emsp; $\ n$ 表示图片的数量，$\ X^k$ 表示第k个图片，$\ y_{true}^k$ 表示第k个图片的正确类别，$\ X_{adv}^k$ 表示第k个图片的对抗样本，函数$\ T$  表示一种任意的图片变换(如打印图片后拍摄照片等)。
$$
C(X,y) = \left\{
\begin{array}{lr}
1\ \  \text{if image X is classified as y}\\
0\ \  \text{otherwise}
\end{array}
\right.
$$
 &emsp; &emsp; 这个公式的意思就是，在被攻击成功的图片里面，可以通过任意的变换使其不被攻击成功的图片的比例。

<br>

### 2、实验部分

 &emsp; &emsp; 作者进行了photo transformation，即：先将图片与对抗样本打印出来，然后拍照，分别得到转换后的clean image和对抗样本。

 &emsp; &emsp; 注：实验获得的所有照片都是手动拍摄，拍摄的角度、光照、距离等都是随机的，这就引入了破环对抗扰动的可能性。

 &emsp; &emsp; 作者进行了两组实验，第一个实验是使用的完整的数据集，即原始图片包含会被正确分类和不会被正确分类的干净数据。第二个实验是使用的预过滤的数据，即原始图片都是被正确分类的干净数据，且对抗样本都被误分类，且置信度都大于0.8。

<br>

### 3、实验结果

 &emsp; &emsp; 实验显示，**FGSM相比于ILLC，对photo变换更加robust。**作者是这样解释的，ILLC使用的是更小的扰动，而这些扰动更容易被photo变换破环。

 &emsp; &emsp; 还有一个没有预料到的结果是，在一些情况下，使用预过滤的数据得到的破坏率比使用完整的数据的高。

 &emsp; &emsp; 实验结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311192916956.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311192925893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)


## 4、Artificial Image Transformations

 &emsp; &emsp; 除了一些自然的变换，作者还是用了一些人工变换进行试验，如：改变亮度和对比度，高斯模糊、高斯噪声，JPEG编码。

 &emsp; &emsp; 作者经过实验发现，FGSM比ILLC更加robust，top-5的破坏率普遍比top-1的高，改变亮度和对比度并不会影响对抗样本很多。

<br>

## 5、Conclusion

 &emsp; &emsp; 在本文中，我们探讨了为在物理世界中运行的机器学习系统创建对抗样本的可能性。我们使用通过手机拍摄的照片作为Inceptionv3网络的输入。我们发现，有相当一部分的对抗样本是成功的。这个发现，也就证明了，在现实世界中的对抗样本是可行的。

 &emsp; &emsp; 今后的工作，我们期待，会有一些使用其他的现实物体的攻击，对抗不同种类的机器学习系统的攻击，例如复杂的强化学习，在那些不能访问模型的参数和结构的网络上的攻击，以及通过明确的为现实世界的变换建模而得到的一个有更高成功率的针对现实世界的攻击



## 6、附录

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311193010335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020031119303677.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311193059239.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)