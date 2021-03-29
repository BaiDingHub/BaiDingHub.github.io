---
title: 对抗样本（十八）PGD及防御
date: 2020-04-03 13:18:05
tags:
 - [深度学习]
 - [对抗攻击]
 - [对抗防御]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,Denfensive PGD"
description: "对抗样本（十八）PGD及防御"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、Towards deep learning models resistant to adversarial attacks.

## 1. Paper Information

> 时间：2017年
>
> 关键词：Adversarial Attack，CV，PGD
>
> 论文位置：https://arxiv.org/pdf/1706.06083)
>
> 引用：Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[J]. arXiv preprint arXiv:1706.06083, 2017.

## 2. Motivation

 &emsp;&emsp; 研究表明，对抗攻击的存在可能是深度学习模型的一个固有弱点。为了解决这个问题，我们从鲁棒性优化的角度研究了神经网络的对抗鲁棒性。这些方法为我们研究这些问题提供了基准。同时，这些方法可以让模型提高对对抗攻击的抵抗力。作者**定义了first-order adversary的概念**作为一种自然和广泛的安全保障。使模型更加鲁棒是走向更好的深度学习模型的重要一步。

 &emsp;&emsp; 在对抗样本领域，我们永远不能确定一个攻击方法找到了最具对抗性的样本，也不能说一个防御方法能够抵御所有的对抗样本，这使得我们很难评估模型的鲁棒性。所以，我们怎么样才能够训练一个更加鲁棒的模型呢?

## 3. Main Arguments

 &emsp;&emsp; 在这篇论文中，我们**从鲁棒性优化的角度研究了神经网络的对抗鲁棒性**，我们**使用了一种natural saddle point (min-max) 公式来定义对对抗攻击的安全性**。这个公式将攻击和防御放到了同一个理论框架中，使得我们对攻击和防御的性能有了良好的量化。

- **我们对此鞍点公式相对应的优化场景进行了仔细的实验研究**，**提出了PGD**这个**一阶方法**（利用局部一阶信息）来解决这个问题。
- 我们**探讨了网络架构对对抗鲁棒性的影响，发现模型容量在其中扮演了重要的角色**。网络通常需要比正常分类情况更大的模型容量，才能够抵御强大的对抗攻击。这表明鞍点问题的鲁棒决策边界比简单地分离正常数据点的决策边界要复杂得多。
- **基于上述鞍点的优化，并使用PGD作为攻击手段，我们在MNIST和CIFAR10上训练出了鲁棒的分类模型**。我们使用了大量的攻击手段进行测试，MNIST模型可以在攻击下保持89%的分类准确率，甚至可以抵御迭代的白盒攻击；CIFAR模型可以保持46%的分类准确率。另外，在黑盒攻击下，MNIST和CIFAR10模型分别可以实现95%和64%的准确率。

## 4. Framework

### 4.1 An Optimization View on Adverarial Robustness

 &emsp;&emsp; 我们的大部分讨论将围绕着**对抗健壮性的优化视角**。这一视角不仅准确地捕捉了我们想要研究的现象，而且还将为我们的调查提供信息。

 &emsp;&emsp; 在训练模型时，我们通常是**通过经验损失最小化（ERM）来训练**的，即最小化$\ \mathbb{E}_{(x,y)\sim \mathcal{D}}[L(x,y,\theta)]$ ，不幸的是，ERM通常并不能够得到鲁棒的模型。那么，为了能够训练出鲁棒的模型，我们需要去**增强ERM过程**。首先，我们提出了一个基准，即一个鲁棒的模型应该满足的条件，即guarantee。之后，我们调整训练方法来满足这个guarantee。

 &emsp;&emsp; 第一步，**我们要先选择一个要攻击的模型**。对于每一个数据点，我们引入了扰动边界$\ \mathcal{S} \in \mathbb{R}^d$ ，即对抗扰动允许的扰动范围。比如，有些方法中扰动边界选择$\ x$ 的$\ l_{\infty}$ 球作为扰动边界。

 &emsp;&emsp; 之后，**我们将上面的扰动因素考虑到训练过程中去**。我们并**不是直接将分布为$\ \mathcal{D}$ 的样本直接送入损失函数中去，而是先允许样本有轻微的扰动**，这就得到了下面的**鞍点问题**，也是我们研究的主要问题，即：
$$
min_{\theta}\rho(\theta),\ \text{where}\ \rho(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}[max_{\delta \in \mathcal{S}}L(\theta,x+\delta,y)]
$$
 &emsp;&emsp; 首先，这个公式为我们提供了一个统一的视角，它包含了以前关于对抗鲁棒性的许多工作。**上面的公式包含一个最大化和一个最小化过程。最大化的意思是找到一个给定数据点$\ x$ 的能使损失函数最大化的对抗版本，这正是攻击问题。最小化的目标是优化模型参数，使得内部的对抗损失最小化，而这正式一个防御问题。**

 &emsp;&emsp; 第二，**这个鞍点公式明确了理想鲁棒分类器应该达到的目标，以及鲁棒分类器鲁棒性的定量度量**。

### 4.2 Towards Universally Robust Networks

 &emsp;&emsp; 有了上面的鞍点公式，我们可以发现，在数据点周围的小的对抗扰动都会无效，因此，我**们把我们的精力集中于如何解决这个问题**。

 &emsp;&emsp; 不幸的是，这个鞍点问题并不是那么容易解决。该问题涉及到出处理一个非凸的外部最小化问题和一个非凹的内部最大化问题。

#### 4.2.1 The Landscape of Adversarial Examples

 &emsp;&emsp; 内部的最大化问题，对应于，给定模型和数据点，找到一个对抗样本的问题。因为，这个问题要求我们最大化一个非凹的函数，人们会认为它很难处理。事实上，**我们可以线性化这个问题，然后进行解决，比如FGSM方法**。不过，FGSM这种单步方法有一些缺点，无法应对复杂的对抗攻击。

 &emsp;&emsp; 为了更好地理解这个问题，我们研究了MNIST和CIFAR10上多个模型的局部最大值。我们主要用到的工具是PGD方法，因为，它是大规模约束优化的标准方法。我们**在数据点的$\ l_{\infty}$ 球边界的许多点运行PGD**，来探索损失函数中的大部分位置（即在这个球的范围内，探索到绝大多数地方，以能够找到最强的对抗样本）。

 &emsp;&emsp; 通过这种方法，我们发现我们可以解决这个问题了。虽然在$\ x_i + \mathcal{S}$ 的范围内有许多局部最大值，但是他们的损失值往往相似，也就是说，解决了这个问题，我们就可以去训练神经网络了。

 &emsp;&emsp; 另外，在实验中，我们发现了一下的现象：

- 我们发现，当我们在$\ x_i + \mathcal{S}$ 内随意选取起始点进行PGD时，**发现损失值以相当一致的方式增加，并且逐渐收敛**

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/1.png?raw=true)



- 对上述问题进一步分析，我们可以发现，在大量的随机启动后，**最终迭代的损失遵循一个良好的集中分布，没有极端异常值**

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/2.png?raw=true" alt="2" style="zoom:50%;" />

 &emsp;&emsp; 上述的观点也说明了PGD是一个通用的攻击方法。

#### 4.2.2 First-Order Adversaries

 &emsp;&emsp; 实验发现，通过PGD发现的局部最大值在正常训练的网络和对抗训练的网络中都有着相似的损失值。这也说明，**只要能够防御住PGD，就会对所有的一阶攻击手段具有鲁棒性**。这在第五节做了实验。

 &emsp;&emsp; 这种鲁棒性会在黑盒攻击下变得更强。我们在附录中讨论了迁移性。我们发现，提**高模型容量和攻击的健壮性，可以提高我们模型对迁移攻击的抵抗能力。**

#### 4.2.3 Descent Directions for Adversarial Training

 &emsp;&emsp; 我们利用PGD很好的解决了内部优化问题，那么下一步就是要解决外部优化问题，即寻找使“对抗损失”最小化的模型参数，即内部最大化问题的值。

 &emsp;&emsp; 而到了这一步就是正常的网络的优化问题，我们可以使用SGD来进行解决。

### 4.3 

 &emsp;&emsp; **只是能解决这个问题还是不够的，我们还需要做出实验证明我们可以使得loss足够的小。**对于一组固定的输入扰动$\ S$ 来说，其loss值由分类器架构决定。而，**要使得分类器更加鲁棒，就需要增加分类器的容量，这是因为，我们的方法使得决策边界变得复杂**，因此需要更大容量的分类器，如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/3.png?raw=true" alt="3" style="zoom:50%;" />

## 5.Result

### 5.1 Network Capacity and Adversarial Robustness

 &emsp;&emsp; 为了验证这个观点，作者进行了实验。

 &emsp;&emsp; 在MNIST数据集上，采用了一个简单的卷积网络，一个2个卷积核的CNN+4个卷积核的CNN+64个单元的FC，每个CNN后跟2x2的max-pooling。作者通过城北的增加CNN的卷积核和FC的单元数来提高模型容量。对抗样本所采用的扰动值$\ \epsilon = 0.3$ 。

 &emsp;&emsp; 在CIFAR10数据集上，使用了ResNet网络，作者使用了裁剪和反转的数据增强方法。为了增强模型容量，在原先模型的基础上增加了10倍的层数，包含了5个残差单元，分别是16、160、320、640卷积核。

 &emsp;&emsp; 实验结果如下，第一张图表示在原始数据上训练，在原始数据、FGSM对抗样本、PGD对抗样本下的分类准确率。第二张图表示借用FGSM对抗样本进行训练（作者所说的方法）下的准确率，第三张图是借用PGD对抗样本进行训练下的准确率。第四张图表示在增强模型容量时loss的下降程度。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/4.png?raw=true" alt="4" style="zoom:50%;" />

- **模型容量影响**：在仅使用正常样本进行训练时，可以发现增强对抗样本可以增加对单步扰动（FGSM）的鲁棒性，而且$\ \epsilon$ 越小，影响越大。
- **FGSM对抗样本训练的研究**：可以发现，在模型容量较小时，模型对FGSM对抗样本产生了过拟合，这种行为被称为label leaking。而且，模型无法防御PGD攻击
- **PGD对抗样本训练的研究**：可以发现，在模型容量较小时，模型无法拟合，而模型较大时就可以有防御效果了。
- **鞍点问题的研究**：可以看到，随着模型容量的增大，loss值也在变小，也就是说增大模型容量可以使模型更好地适应对抗攻击。
- **迁移性研究**：增大模型容量、使用更强的攻击方法，会使模型越不容易受到对抗样本迁移性的攻击。不过，当模型容量增大到一定程度时，关联性就没有那么大了。

### 5.2 Adversarially Robust Deep Learning Models

 &emsp;&emsp; 在有了上面想法后，我们就要训练出一个足够鲁棒的分类器，主要从两方面下手，即**训练一个高容量的网络、使用更强的攻击方法作为辅助**。

 &emsp;&emsp; 对于MNIST和CIFAR10数据集来说，PGD作为攻击手段就已经足够。在使用PGD时，每个epoch，每轮样本只扰动一次，等下次epoch再扰动。

#### 5.2.1 MNIST

 &emsp;&emsp; 在MNIST数据集上，作者使用40迭代次数的PGD作为攻击方法，step size设为0.01，使用的是梯度的符号。模型由两个CNN和一个FC组成，CNN的卷积核数分别是32和64，每一个CNN后跟2x2的max pooling，之后接1024的FC。该模型在不同的攻击下的效果如下，其中A表示白盒攻击，A‘表示黑盒攻击（从一个完全相同的网络中迁移，不过不是同一个网络），$\ A_{nat}$ 表示黑盒攻击（从同一个网络中迁移，仅使用原始样本训练），B表示黑盒统计（从不同架构的网络中迁移），：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/5.png?raw=true" alt="5" style="zoom:50%;" />

#### 5.2.2 CIFAR10

 &emsp;&emsp; 对于CIFAR10，考虑了两个模型结构，原始ResNet和10倍大的变体。step为7，size为2。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/6.png?raw=true" alt="6" style="zoom:50%;" />

#### 5.2.3 不同$\ \epsilon$ 值的影响

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%85%AB%EF%BC%89PGD%E5%8F%8A%E9%98%B2%E5%BE%A1/7.png?raw=true" alt="7" style="zoom:50%;" />

## 6. Argument

 &emsp;&emsp; 该论文提出了PGD攻击方法，使用迭代多步的方法寻找对抗样本，比FGSM这种单步方法要强，同时设计了新的损失函数，用来增加模型的鲁棒性，取得了不错的成果。

## 7. Further research

 &emsp;&emsp; PGD是类似SGD那样的迭代方法，我们可以对其进行扩展，使用Momentum等方法来进行迭代，衍生了一下论文，Boosting adversarial attacks with momentum。

