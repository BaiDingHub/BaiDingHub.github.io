---
title: 对抗样本（十五）Natural GAN
date: 2020-04-03 13:15:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,Natural GAN"
description: "对抗样本（十五）Natural GAN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **Generating natural adversarial examples** 

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2017年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp; https://arxiv.org/abs/1710.11342

<br>

# 二、论文背景及简介

 &emsp;&emsp; 目前，对抗扰动都很不自然、没有语义意义、无法应用到复杂的领域，例如语言。本文利用生成对抗网络的最新进展，提出了一种**在数据流形上生成自然易读的对抗样本**的方法，该方法通过搜索密集连续数据表示的语义空间来生成自然易读的对抗实例。

# 三、论文内容总结

- 借用GAN，生成更加自然的对抗样本

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

# 四、论文主要内容

## 1. Introduction

<p style="text-indent:3em">尽管对抗样本揭露了ML存在的盲点问题，但是这些对抗样本并不会在自然情况下产生，在模型部署时，很难遇到这样的样本。因此，很难深入了解黑盒分类中的基本决策行为：为什么对于对抗样本而言决策是不同的？为了阻止这一行为，我们改变了什么？分类器对数据的自然变化是不是鲁棒的？另外，在输入空间和语义空间存在不匹配的问题，我们对输入空间做一些无意义的变化比如光照强度的变化，却会导致其在语义空间发生大的变化。</p>

<p style="text-indent:3em">最直接的想法就是对于攻击者而言，在一个密集的连续的数据表示空间中搜索，而不是在直接在数据空间中搜索。我们使用GAN将正态分布的固定长度向量映射到数据样本。在给定输入实例的情况下，我们通过在递归收紧的范围内采样，在潜在空间中的对应表示附近搜索对抗样本。</p>

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

## 2. Framework For Generating Natural Adversaries

 &emsp;&emsp; 给定黑盒分类器$\ f$ ，一组未标签数据$\ X$ ，目标是为一个给定的样本$\ x$ 生成对抗样本$\ x^*$ ，注意这里的$\ x$ 不一定在$\ X$ 内，但一定与$\ X$ 来自于同样的数据分布$\ P_x$ 。我们想要**基于数据分布$\ P_x$ 的流行上，让$\ x^*$ 离$\ x$ 最近**，而不是在原始的数据表示上。

 &emsp;&emsp; 而距离的评判并不是在原始的输入空间，我们将会在一个对应的密集表示空间$\ z$ 中来寻找$\ x^*$ 。也就是说，**我们首先会有一个现在的密集向量空间，其与数据分布$\ P_x$ 对应，我们将原始样本映射到空间中，在其附近找到对抗样本$\ z^*$ ，使用GAN映射回原始空间，得到最终的对抗样本$\ x^*$ 。** 通过这种方法得到的对抗样本会更加的有效（图片更加清晰，句子更有语法），在于以上更接近于原始输入。

### 2.1 GAN介绍

 &emsp;&emsp; 为了解决上述问题，我们需要**一个强大的生成模型来学习从潜在的低维表示到分布$\ P_x$ 的映射**，我们使用$\ X$ 中的样本进行估计。

 &emsp;&emsp; 给定未标签的数据集$\ X$ 作为训练数据**，生成器会把分布为$\ p_z(z)，z\in \mathbb{IR}^d$ 的噪声映射到尽可能接近训练数据的合成数据上。判别器来区分生成器的输出与来自$\ X$ 的真实样本的输出。**在这里，为了优化的方便，采用了**WGAN**，其目标函数为：
$$
min_\theta\ max_w\ \mathbb{E}_{x\sim p_x(x)}[C_w(x)] - \mathbb{E}_{z\sim p_z(z)}[C_w(G_\theta(z))]
$$

### 2.2 Natural Adversaries

 &emsp;&emsp; 为了表示一些自然的样本，我们首先训练一个WGAN，其生成器用来将随机向量$\ z\in \mathbb{IR}^d$ 映射到$\ X$ 的某一样本$\ x$ 。我们也训练了一个逆变器$\ I_{\gamma}$ ，用来映射数据的距离到相应的密集表示向量。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

 &emsp;&emsp; 我们最小化$\ x$ 的重建误差和$\ z$ 与$\ I_\gamma(G_\theta(z))$ 的差别来促使隐藏空间变得正态分布：
$$
min_\gamma\ \mathbb{E}_{x\sim p_x(x)}||G_\theta(I_\gamma(x))-x||+\lambda· \mathbb{E}_{z\sim p_z(z)}[L(z,I_\gamma(G_\theta(z)))]
$$
 &emsp;&emsp; 通过这些学到的函数，我们可以如下定义对抗样本$\ x^*$ ：
$$
x^* = G_\theta(z^*)\ where\ z^* = argmin_{\tilde{z}}||\tilde{z}-I_\gamma(x)|| \\
s.t. \ f(G_\theta(\tilde{z})) \ne f(x)
$$
  &emsp;&emsp; 作者在图片领域采用$\ L_2$ 距离，$\ \lambda=0.1$ ；在文本领域采用Jensen-Shannon距离，$\ \lambda=1$。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

### 2.3 Search Algorithms

 &emsp;&emsp; 作者提出了两种方法在$\ z'$ 得附近寻找扰动$\ \tilde{z}$ 。

#### ① 随机迭代寻找

 &emsp;&emsp; 我们会递增的扩大寻找范围$\ \Delta r$ ，在该范围内，随机采样$\ N$ 个，知道我们生成了对抗样本$\ \tilde{x}$ 。在这些对抗样本中，我们会选择最接近$\ z^*$ 得那一个作为输出。

#### ② 混合收缩寻找

 &emsp;&emsp; 为了提高第一种算法的效果，提出了一种从粗到细得策略。我们首先在较宽的搜索范围内搜索，然后在二等分中以更密集的抽样递归地收缩搜索范围的上限。其速度大概是第一种方法得四倍。

## 3.Illustrative Examples

### 3.1 Generating Image Adversaries

 &emsp;&emsp; 采用数据集MNIST和LSUN，使用参数$\ \Delta r=0.01,N=5000$

#### 3.1.1 Handwritten Digits

 &emsp;&emsp; 在MNIST数据集上，训练了一个$\ z\in \mathbb{IR}^{64}$ 得WGAN。生成器由转置卷积组成，判别器由卷积层组成，在判别器的最后一层隐藏层上添加全连接层构建逆变器。

 &emsp;&emsp; 采用的目标分类器有两个，一个是Random Forest，五棵树，测试准确率90.45%；一个是LeNet，测试准确率98.71%。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

#### 3.1.2 Church vs Tower

 &emsp;&emsp; 在LSUN中取126227张church和tower的图片，resize到64x64。训练了一个$\ z\in \mathbb{IR}^{128}$ 的WGAN，生成器和判别器采用了残差网络。采用MLP分类器，在这两类的测试准确率为71.3%。

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

### 3.2 Generating Text Adversaries

 &emsp;&emsp; 略

<br>

## 4.Experiments

### 4.1 Robustness of Black-box Classifier

 &emsp;&emsp; 将我们的方法用到各种各样的黑盒分类其中，来评测这些模型的鲁棒性。一般来说，更精确的分类器需要对样本进行更多的改变才能改变其预测值。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

### 4.2 Human Evaluation

 &emsp;&emsp; 作者进行了调查，询问生成的对抗样本的自然程度和与原始样本的相似程度

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%94%EF%BC%89Natural%20GAN/1.png?raw=true)

## 5. Related Work

 &emsp;&emsp; 略

## 6. Discussion And Future Work

- 提高GAN模型的能力，找到能处理文本离散问题的模型

## 7. Conclusions

 &emsp;&emsp; 本文，提出了一个生成更加自然的对抗样本的方法，将这种方法应用到了视觉和文本领域。