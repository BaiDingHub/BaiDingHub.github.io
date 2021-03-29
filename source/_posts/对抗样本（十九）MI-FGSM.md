---
title: 对抗样本（十九）MI-FGSM
date: 2020-04-03 13:19:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,MI-FGSM"
description: "对抗样本（十九）MI-FGSM"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

## 1. Paper Information

> 时间：2018年
>
> 关键词：Adversarial Attack，CV，Momentum，MI-FGSM
>
> 论文位置：https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf
>
> 引用：Dong Y, Liao F, Pang T, et al. Boosting adversarial attacks with momentum[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.

## 2. Motivation

 &emsp;&emsp; 虽然，如今在白盒攻击下有很多种攻击方法都取得了不错的效果，比如L-BFGS、FGSM、BIM等，但是大多数的对抗攻击在黑盒攻击下只有比较低的成功率，也就是说，**这些攻击的迁移性并不好**。通常来说，**攻击能力和迁移性之间存在trade-off**，基于优化的和迭代的攻击方法往往有较强的攻击能力，但迁移性较差；基于单步梯度的攻击方法虽然迁移性较强，但是白盒攻击时攻击能力较低，那么黑盒攻击时效果也不会很好。为了解决这个问题，作者提出了一种基**于Momentum迭代的对抗攻击方法**。

## 3. Main Arguments

 &emsp;&emsp; 作者将Momentum元素加入到了攻击的迭代步骤中，该方法可以稳定更新方向，避免迭代过程中的局部极值，而且我们发现，该方法在黑盒攻击和白盒攻击中都有较高的成功率。使用这种方法，作者赢得了NIPS2017非目标对抗攻击和目标攻击竞赛的第一名。作者**将Momentum与迭代FGSM相结合，提出了momentum iterative fast gradient sign method (MI-FGSM)**，用于在非目标攻击时生成满足$\ L_{\infty}$ 范数约束的对抗样本。之后，**提出了几种有效的攻击集成模型的方法**。最后，**作者将MI-FGSM拓展到$\ L_2$ 范数约束和目标攻击**。

## 4. Framework

### 4.1 Momentum iterative fast gradient sign method

 &emsp;&emsp; 对抗样本生成的优化目标为：
$$
\arg \max_{x^*}\ J(x^*,y), \ \ \ s.t. ||x^* - x||_{\infty} \le \epsilon
$$
 &emsp;&emsp; FGSM方法将决策边界设想为线性的，但一般决策边界都不是线性的。基于迭代的攻击方法惠子每次迭代时贪婪的将对抗样本往梯度方向移动，使得对抗样本容易陷入局部极值，造成模型的过拟合，使得迁移性下降。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/1.png?raw=true" alt="1" style="zoom: 67%;" />

### 4.2 Attacking ensemble of models

 &emsp;&emsp; 为了提高模型的性能和鲁棒性，集成模型在研究和竞赛中得到了广泛的应用。而集成的思想也可以用在对抗攻击中，如果一个样本能够对多个模型保持对抗特性，那么相当于该样本具有一个对抗的内在方向，可以欺骗这些模型，也就是说，该样本就更具有迁移性，在黑盒攻击时就更有效果。

 &emsp;&emsp; 作者提出了一种利用多个模型的方法，即将多个模型的logit融合在一起，我们把它称之为ensemble in logits。为了攻击K个模型的融合模型，我们将logits进行融合：
$$
l(x) = \sum_{k=1}^K w_k l_k(x)
$$
 &emsp;&emsp; 其中$\ l_k(x)$ 是第k个模型的logits，$\ w_k$ 是模型权重，满足$\ \sum_{k=1}^{K} w_k = 1$ 。损失函数使用softmax损失，即：
$$
J(x,y) = -1_y·\log(\text{softmax}(l(x)))
$$
   &emsp;&emsp; 其中，$\ 1_y$ 表示y的one-hot形式。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/2.png?raw=true" alt="2" style="zoom: 67%;" />

 &emsp;&emsp; 除了这一种集成策略，作者还介绍了其他的集成策略：

- 一种是将K个模型的概率输出继承，即$\ p(x) = \sum_{k=1}^K w_k p_k(x)$ 
- 第二种是将K个模型的loss集成在一起，即$\ J(x,y) = \sum_{k=1}^K w_k J_k(x,y)$ 

 &emsp;&emsp; 作者认为，logits的集成优于这两种策略。这在接下来会被证明。

### 4.3 Extensions

#### 4.3.1 $\ L_2 $ 范数

 &emsp;&emsp; 如果采用L2范数的话，我们要使用MI-FGM方法，即将上式改为：
$$
x_{t+1}^* = x_t^* + \alpha · \frac{g_{t+1}}{||g_{t+1}||_2}
$$

#### 4.3.2 目标攻击

 &emsp;&emsp; 对于目标攻击，要修改两个部分：
$$
g_{t+1} = \mu · g_t + \frac{J(x_t^*,y^*)}{||\nabla_x J(x_t^*,y^*)||_1} \\
x_{t+1}^* = x_t^* - \alpha · sign(g_{t+1})
$$

## 5.Result

### 5.1 Setup

 &emsp;&emsp; 作者使用了ImageNet数据集，作者研究了七个模型，其中四个是正常训练的模型，Inception v3 (Inc-v3) , Inception v4 (Inc-v4), Inception Resnet v2  (IncRes-v2) , Resnet v2-152 (Res-152)。其他三个是集成对抗训练的模型，$\ \text{Inc-v3_{ens3}}$ 、$\ \text{Inc-v3_{ens4}}$ 、$\ \text{IncRes-v2_{ens}}$ 。

 &emsp;&emsp; 如果模型不能对样本正确分类，那么研究攻击是没有意义的。所以作者从ILSVRC 2012的验证集中选择了1000个图片，属于1000各类别，这些图片都是被正确分类的。

### 5.2 Attacking a single model

 &emsp;&emsp; 作者使用了FGSM，迭代的FGSM，以及MI-FGSM进行实验。选择的扰动大小为$\ \epsilon = 16$ ，迭代次数选择为10。MI-FGSM的decay设为$\ \mu = 0.1$ 。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/3.png?raw=true)

 &emsp;&emsp; 可以看到，I-FGSM在黑盒攻击时效果一般不如FGSM，但是MI-FGSM在黑盒攻击时效果很好，而且在白盒攻击时效果也特别好。不过，虽然MI-FGSM极大的提高了黑盒攻击的成功率，但是它仍然无法有效的攻击经过集成训练的模型，比如$\ \text{IncRes-v2_{ens}}$ ，成功率小于16%。

#### 5.2.1 Decay factor $\ \mu$ 

 &emsp;&emsp; Decay factor起到了重要的作用，因此在这研究该因素的影响。我们使用由Inc-v3模型生成的对抗样本进行白盒、黑盒攻击，采用扰动值$\ \epsilon = 16$ ，采用迭代次数10，研究Decay factor在0到2之间的影响。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/4.png?raw=true" alt="4" style="zoom: 67%;" />

 &emsp;&emsp; 可以看到，在$\ \mu = 1$ 时黑盒攻击效果最好，也就是简单的将过去的梯度完全相加。

#### 5.2.2 The number of iterations

 &emsp;&emsp; 这一节研究迭代次数的影响，设置$\ \epsilon = 16,\mu=1.0$ ，使用由Inc-v3模型生成的对抗样本进行白盒、黑盒攻击。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/5.png?raw=true" alt="5" style="zoom:67%;" />

 &emsp;&emsp; 可以看出，在黑盒攻击时，I-FGSM的成功率随迭代次数增加而下降，但是MI-FGSM保持一个较高的值。

#### 5.2.3 Update directions

 &emsp;&emsp; 为了研究为什么MI-FGSM有更好的迁移性，我们研究了I-FGSM和MI-FGSM的梯度更新方向，我们计算了两个扰动的余弦相似度：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/6.png?raw=trueg" alt="6" style="zoom:67%;" />

 &emsp;&emsp; 我们可以看出，MI-FGSM的余弦相似度更大，也就是说MI-FGSM的更新方向比I-FGSM更稳定。而迁移性的存在是因为不同的模型往往学习相似的决策边界，不过，虽然决策边界相似，但由于DNN的高度非线性结构，所以不同模型的决策边界往往是不同的。所以，**某个模型在某一数据点附近可能存在一些异常的决策区域，这些区域很难迁移到其他模型中，而这些区域存在局部极值，所以I-FGSM很容易陷入这些区域，导致迁移性较差，而MI-FGSM可以逃离这些区域，使得迁移性更好。**

#### 5.2.4 The size of perturbation

 &emsp;&emsp; 作者最后研究了不同的扰动大小的影响。设置I-FGSM和MI-FGSM的step size$\ \alpha = 1$ ，所以迭代次数增加，那么扰动大小就会增加。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/7.png?raw=true" alt="7" style="zoom:67%;" />

 &emsp;&emsp; 可以看到，FGSM在扰动值较大时攻击成功率也较低，这也是因为FGSM的线性决策边界假设造成的。在黑盒攻击下，可以看出MI-FGSM随扰动值增加的最快。

### 5.3 Attacking an ensemble of models

 &emsp;&emsp; 在这一节比较了不同的集成策略对攻击的影响。

#### 5.3.1 Comparison of ensemble methods

 &emsp;&emsp; 我们使用了四个模型，Inc-v3, Inc-v4, IncRes-v2 and Res-152.在实验时，我们将一个模型设为留出的黑盒模型，使用另外三个模型集成，然后对他进行攻击，使用了三种集成策略以及三种攻击方法。同时，使用四个模型集成，即白盒攻击，作为对比，实验结果如下：

![8](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/8.png?raw=true)

 &emsp;&emsp; 我们可以看到，logits的集成策略优于其他两种策略。

#### 5.3.2 Attacking adversarially trained models

 &emsp;&emsp; 接下来，我们来看看对对抗训练后的模型的攻击效果。我们使用一个模型留出，使用另外六个模型logits集成进行黑盒攻击（Hold-out），以及使用七个模型集成进行白盒攻击（Ensemble），得到实验结果如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%B9%9D%EF%BC%89MI-FGSM/9.png?raw=true" alt="9" style="zoom:67%;" />

## 6. Argument

 &emsp;&emsp; 作者将momentum融入到I-FGSM中，提出了MI-FGSM，这使得模型获得了更强的迁移性。同时，作者提出了一种新的对抗攻击的融合策略，利用这种方法又大大提高了黑盒攻击的攻击能力。

## 7. Further research

