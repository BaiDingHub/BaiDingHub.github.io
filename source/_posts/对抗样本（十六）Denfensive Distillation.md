---
title: 对抗样本（十六）Denfensive Distillation
date: 2020-04-03 13:16:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,Denfensive Distillation"
description: "对抗样本（十六）Denfensive Distillation"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks** 

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2016年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp; https://arxiv.org/abs/1511.04508

<br>

# 二、论文背景及简介

 &emsp;&emsp; 为了防御之前提出的FGSM和JSMA的攻击方式，作者根据之前hinton提出的蒸馏学习的方式，在此基础上稍作修改得到了防御蒸馏模型，并理论推导了防御有效性的原因。其实验证明，防御性蒸馏将对抗性样本的成功率与在MNIST数据集上训练的原始DNN相比，从95.89％降低到0.45％，而对在CIFAR10上训练的产原始DNN模型则从87.89％降低到5.11％。对蒸馏参数空间的进一步经验探索表明，正确的参数化可以将DNN对输入扰动的敏感性降低$\ 10^{30}$倍。还发现，蒸馏增加了在生成对抗样本时，需要修改的平均最小特征数，平均提升了800%。

# 三、论文内容总结

- 阐明了防御对抗样本的设计要求，强调了防御鲁棒性、输出精度、DNN性能之间的内在矛盾
- 介绍了Denfensive Distillation防御方法，可以使DNN模型对扰动更加鲁棒。与之前的蒸馏目的不同，本文方法是将获取的知识反馈到原始模型中。
- 防御蒸馏通过降低对输入扰动的敏感性来生成更加平滑的分类器，这些分类器对对抗样本更有弹性，有更强的泛化性。

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

# 四、论文主要内容

## 1. Introduction

 &emsp;&emsp; Distillation原本被用来将一个DNN的knowledge迁移到另一个不同的网络。Distillation可以从大网络迁移到小网络，从而降低了DNN结构的复杂性。我们构建了一个Distillation的变种，来进行对抗防御。我们的目的是使用从DNN提取到的knowledge来提升它自己对对抗样本的鲁棒性。

 &emsp;&emsp; 如果对抗性梯度很高，那么制作对抗样本也会变得很容易，因为一个小的扰动就会引起DNN输出的较大的变化。为了防御这样的扰动，我们必须要降低输入的变化而引起的输出的变化。换句话说，我们要使用Denfensive Distillation来平滑模型，使得模型能够在他的训练集之外依旧泛化的很好。

## 2、Adversarial Deep Learing

 &emsp;&emsp; 略

## 3、Denfending DNNs Using Distillation

 &emsp;&emsp; 这一节描述我们怎样将Distillation转换成Defensive Distillation来解决DNN的脆弱性问题。

### 3.1 Defending againt Adversarial Pertubations

 &emsp;&emsp; 为了形式化对对抗防御的讨论，提出了一个评估DNN对对抗噪声的适应力的评价标准，即：网络的鲁棒性。

#### 3.1.1 DNN Robustness

 &emsp;&emsp; DNN对对抗扰动的鲁棒性表示的是对扰动的适应能力。一个鲁棒的DNN应该是（1）对训练集外的数据仍然有较高的准确性；（2）建立一个光滑的分类器函数F，它可以直观地在给定样本的邻域内对输入进行相对一致的分类。

#### 3.1.2 Defense Requirements

 &emsp;&emsp; 对抗防御的要求如下：

- 对网络结构影响小
- 保持网络的准确率
- 保持网络测试时的计算速度
- 防御措施应该适用于相对接近训练数据集中点的对抗性样本

### 3.2 Distillation as a Defense

 &emsp;&emsp; Denensive Distillation与网络蒸馏的不同点在于，我们保留了原有的网络结构。我们的目的是为了恢复而不是压缩。Denfensive Distillation的训练步骤如下：

1. 记数据集为$\ \mathcal{X}$ ，样本$\ X \in \mathcal{X}$ ，其one hot标签为$\ Y(X)$
2. 用训练集$\ \{(X,Y(X))\}$ 训练一个网络$\ F$ ，其温度为$\ T$ 。其softmax输出为$\ F(X)$ ，表示的是所有类的概率向量。记$\ F_i(X)$ 表示其输出的第$\ i$ 个分量。
3. 构建新的训练集$\ \{(X,F(X))\}$ 
4. 使用新的训练集训练另一个网络$\ F^d$ ，其与$\ F$ 采用相同的网络，且温度依旧采用$\ T$ 。这个模型被称为distilled model。注意，预测时采用T=1。

 &emsp;&emsp; 这样的训练方法，可以让模型对数据不会过拟合，会得到一个泛化性更好的网络。高的温度会使网络为每一个类别生成高的概率值。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/1.png?raw=true)

## 4、Analysis Of Denfensive Distillation

- 训练时高温，预测时T=1，使得模型敏感性降低。
- 概率值做标签使得模型学习到了样本之间的结构相似性

## 5、Evaluation

 &emsp;&emsp; 实验部分采用了两个网络。

**三个问题**：

- 防御蒸馏是否**在保持准确性的前提下，提升了网络对对抗扰动的防御力**？

  答：蒸馏将第一个网络的攻击成功率从95.89%降低到了0.45%，将第二个网络的攻击成功率从87.89%降低到了5.11%。准确率只比原本的网络低1.37%。

- 防御蒸馏是否能**降低DNN对输入的敏感性**

  答：防御蒸馏降低了对抗梯度$\ 10^{30}$ 倍，大大降低了敏感性。

- 防御蒸馏是否**让DNN更鲁棒**

  答：防御蒸馏提高了第一个网络的鲁棒性790%，第二个网络的鲁棒性556%（指的是目标攻击的最小平均扰动值）

**数据集采用**：MNIST和CIFAR10。每个数据集对应一个网络结构。MNIST实现了99.51%的准确率，CIFAR10实现了80.95%的准确率。

**攻击策略**：JSMA

**温度值**：T=20，蒸馏后，MNIST准确率99.05%，CIFAR为81.39%。在测试时，T=1。

**不同训练温度下的攻击成功率**

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/2.png?raw=true)

**不同训练温度下的准确率**：

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/3.png?raw=true)

**不同训练温度下对抗梯度的大致分布**：

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/4.png?raw=true)

**不同训练温度下的鲁棒性**：

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AD%EF%BC%89Denfensive%20Distillation/5.png?raw=true)

## 6、Discussion

 &emsp;&emsp; 防御蒸馏的一个限制就是，这只能用在能产生概率分布向量的DNN模型中。