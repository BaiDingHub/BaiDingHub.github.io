---
title: 对抗样本（十二）One Pixel Attack
date: 2020-04-03 13:12:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,One Pixel Attack"
description: "对抗样本（十二）One Pixel Attack"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **One pixel attack for fooling deep neural networks** 

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2017年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp; https://arxiv.org/abs/1710.08864

<br>

# 二、论文背景及简介

 &emsp;&emsp; 在这篇文论中，作者提出了一个在极端有限环境下的一种攻击方式，**只需要修改图片的一个像素**。作者提出了一个方法，利用**进化算法differential evolution(DE)**来生成这样对抗扰动。这种攻击方式属于**黑盒攻击**，通过DE得到的内在特征，它仅仅需要知道很少的信息，就能够欺骗很多类型的网络。

 &emsp;&emsp; 通过实验能得到，该方法能够在Kaggle上的CIFAR-10上攻击67.97%的图像，平均置信度74.03%，能够在ImageNet上攻击16.04%的图像，平均置信度22.91%。

<br>

# 三、论文内容总结

- 论文提出了一种极其特别的攻击方法，**one-pixel attack**，该攻击方式只需要修改图片的一个像素
- 使用**DE进化算法**来得到合适的扰动位置和扰动值
- 在CIFAR-10和ImageNet上做出了实验。

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

# 四、论文主要内容

## 1. Introduction

 &emsp;&emsp; 大多数先前的攻击都没有考虑对抗性攻击的极端有限的场景，即修改可能过多（即修改像素的数量相当大），以致于人眼可以察觉到它。此外，研究在极其有限的场景下产生的对抗性图像可能会对DNN模型在高维空间中的几何特征和整体行为提供新的见解。比如：靠近决策边界的扰动的特征可用来描述边界的形状。

 &emsp;&emsp; 在这篇论文中，通过DE（进化算法）来仅仅扰动一个像素，我们实现了一个黑盒攻击方法。我们的方法的优点如下：

- **有效性**。在Kaggle 的CIFAR-10数据集上，我们通过仅修改一个像素来进行无目标攻击，对三个网络结构进行攻击，达到了68.71%，71.66%，63.53%的成功率。另外，我们发现，每一张原始图像可以平均被扰动成1.8，2.1，1.5个其他的类。在原始的CIFAR-10数据集上，我们有22.6%，35.2%，31.4%的成功率。在ImageNet数据集上，对AlexNet进行无目标攻击，实现了16.04%的成功率。
- **半黑盒攻击**。我们的攻击，仅仅需要黑盒模型的类别概率标签，不需要目标模型的其他信息(梯度、网络结构)。我们的方法比现存的方法更加**简单**，因为我们并不需要把搜索扰动的问题抽象成任何显示的目标函数，**只需要关注能否增加目标类别的概率即可**。
- **灵活性**。我们的攻击可以很多类型的网络，甚至那些不可导的模型。

 &emsp;&emsp; 之所以考虑极其有限的单像素攻击场景，主要有以下几个原因：

- **原始图像的邻域分析**。先前的工作**通过限制扰动的长度来分析原始图像的邻域**，比如universal perturbation的方法。另一方面，**对少量的像素点进行扰动可以看作为通过使用非常低维的切片切割输入空间**，这是一种**探索高维DNN输入空间特征**的另一种方式。one-pixel attack就是上面的攻击方式的一种极端的例子。理论上，它可以给CNN输入空间的理解提供几何学上的见解。
- **A Measure of Perceptiveness**，实际上，这种攻击方法可以有效的隐藏扰动。从之前的论文来看，没有哪一个工作可以证明一个扰动可以完全的被忽略。一个直接得减轻这个问题的方式就是限制扰动的数量。特别的，相比于在理论上提出一种限制条件或者考虑更加复杂的损失函数来限制扰动，我们提出了一个经验解决方案，通过限制可以修改的像素数。换句话说，就是我们**使用像素数作为单位而不是扰动向量的长度来测量扰动强度**，并考虑最坏的情况，即一个像素修改，以及两个其他场景（即3和5个像素）进行比较。

## 2. Related Works

 &emsp;&emsp; 讲解了以下近年来的对抗攻击、对抗防御的工作，简单介绍了对DNN的分类空间描述方便的工作，以及黑盒攻击的工作

## 3. Methodology

### A. Problem Descroption

 &emsp;&emsp; 令$\ f$ 表示目标分类器，其输入为$\ x=(x_1,...,x_n)$ ，原始标签为$\ t$ ，在类别$\ t$ 处的概率为$\ f_t(x)$ ，扰动为$\ e(x)=(e_1,...,e_n)$ ，目标类别为$\ adv$ ，其最大限制值为$\ L$ 。则生成对抗样本的优化问题描述为：
$$
\begin{equation}
\begin{split}
max_{e(x)^*} &\ f_{adv}(x+e(x))\\
s.t.\quad&||e(x)|| \le L\\
\end{split}
\end{equation}
$$
 &emsp;&emsp; 该问题主要是去寻找两个值：1.需要扰动哪个维度的值；2.每个维度需要扰动多大的值。

 &emsp;&emsp; 对于我们这个问题来说，我们需要对其进行一点点的修改：
$$
\begin{equation}
\begin{split}
max_{e(x)^*} &\ f_{adv}(x+e(x))\\
s.t.\quad&||e(x)||_0 \le d\\
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中$\ d$ 是一个比较小的值，在one-pixel attack中$\ d=1$ 。

 &emsp;&emsp; one-pixel attack可以被看作是在一个数据点的n维空间中的一维的方向上进行移动

### B. Differential Evolution DE进化算法

 &emsp;&emsp; DE进化算法是一种基决复杂多模态优化问题的流行的优化算法。DE属于进化算法的一般范畴。此外，它**在种群选择阶段有保持多样性的机制**，因此在实践中，它有望**有效地找到比基于梯度的解甚至其他类型的进化算法更高质量的解**。

 &emsp;&emsp; 具体来说，在每次迭代过程中，根据当前总体（父项）生成另一组候选解决方案（子项）。然后将这些孩子与他们相应的父母进行比较，如果他们比他们的父母更适合（拥有更高的适应值），他们就可以存活下来。这样，只有将父母和孩子进行比较，才能同时达到保持多样性和提高适应值的目的。

 &emsp;&emsp; DE算法并**不使用梯度信息来进行优化**，因此不要求目标函数是可微的或以前已知的。因此，与基于梯度的方法相比，它可以用于更广泛的优化问题（例如，不可微、动态、噪声等）。使用DE生成对抗样本具有以下主要优点：

- **Higher probability of Finding Global Optima 找到全局最优解的概率较高**。DE是一种元启发式算法，与梯度下降或贪婪搜索算法相比，它相对较少受到局部极小的影响（这部分是由于多样性保持机制和一组候选解的使用）。此外，本文考虑的问题有一个严格的约束（只能修改一个像素），这使得它相对困难。
- **Require Less Information from Target System 需要较少的目标系统的信息**。DE不要求优化问题如梯度下降法、拟牛顿法等经典优化方法所要求的那样是可微的。这在生成敌对图像的情况下是至关重要的，因为，1）有些网络是不可微的。2） 计算梯度需要更多关于目标系统的信息，这在很多情况下是不现实的。
- **Simplicity 简单**。这里提出的方法与使用的分类器无关。要使攻击发生，只需知道概率标签就足够了。

 &emsp;&emsp; 有许多对DE算法的变体或改进，如自适应、多目标等。考虑到这些变化/改进，当前的工作可以进一步改进。

### C. Method and Settings 方法 

 &emsp;&emsp; 我们将扰动编码成一个矩阵（候选解），矩阵通过差分进化进行优化（进化）。一个候选解包含固定数量的扰动，每个扰动是一个包含五个元素的元组：x-y坐标和扰动的RGB值。一个扰动修改一个像素。候选解（总体）的初始数目为400，在每次迭代中，将使用通常的DE公式生成另外400个候选解（子解）：
$$
\begin{equation}
\begin{split}
x_i(g+1) =& x_{r_1}(g) + F(x_{r_2}(g) - x_{r_3}(g))\\
&r_1 \ne r_2 \ne r_3\\
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中$\ x_i$ 是候选解的某个元素，$\ r_1,r_2,r_3$ 表示随机数，$\ F$ 是一个范围参数，设为0.5，$\ g$ 表示目前迭代的index。 

 &emsp;&emsp; 在每一次迭代后，我们会将每一个候选解，与其相对应的父解进行对比，胜者进入下一轮迭代过程。

 &emsp;&emsp; 最大迭代次数设置为100，当对Kaggle-CIFAR-10进行有目标攻击时，当目标类的概率标签超过90%时或当ImageNet受到无目标攻击时，当真类的标签低于5%时，会进行early stop。之后，类别真实标签与最高概率的非真实标签进行比较来评价是否该攻击成功了。

 &emsp;&emsp; **初始化父解**，通过对CIFAR-10图像使用均匀分布U（1，32）和对ImageNet图像使用均匀分布U（1，227）来初始化，以生成x-y坐标。通过高斯分布$\ N(\mu=128,\sigma=127)$ 来初始化RGB值。

 &emsp;&emsp; fitness函数，在CIFAR-10上采用目标类别的概率，在ImageNet上采用真实类别的概率。

## 4. Evaluation And Results

 &emsp;&emsp; 在CIFAR-10和ImageNet上对上述攻击方法进行评估，引入了几种对攻击效率的评价标准：

- 攻击成功率
- 对抗概率标签（置信度）：累积每个成功扰动的目标类的概率标签值，然后除以成功扰动的总数。该度量表示当错误分类敌对图像时，目标系统给出的平均置信度
- 目标类别数量：统计成功扰动到目标类的特定数量（即从0到9）的自然图像的数量。特别是，通过计算不能被任何其他类干扰的图像数量，可以评估非目标攻击的有效性。
- 原始目标类对的数量：统计每个原始目标类对被攻击成功的次数。

### A. Kaggle CIFAR-10

 &emsp;&emsp; 在CIFAR-10训练集上，作者训练了3种类型的网络，分别时全卷积网络，NIN网络和VGG16网络，其模型结构如下：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/1.png?raw=true)

 &emsp;&emsp; 对每一个网络都进行有目标攻击和无目标攻击，每一次攻击都从Kaggle CIFAR-10的测试集中随机选取500张图片。

 &emsp;&emsp; 注意，我们使用Kaggle CIFAR-10测试数据集而不是最初的实验。该数据集包含300000幅CIFAR-10图像，这些图像可以进行如下修改：复制、旋转、剪切、模糊、添加少量随机坏像素等。然而，具体采用的修改算法尚未发布。这使得它成为一个更实用的数据集，可以模拟图像可能包含未知随机噪声的常见场景。

 &emsp;&emsp; 此外，在全卷积网络上进行了一个实验，生成了500幅经过3和5个像素修改的对抗性图像。其目的是比较单像素攻击与三像素和五像素攻击。对于每个自然图像，会进行9次目标攻击，试图将其扰动到其他9个目标类。需要注意的是，我们实际上只是进行有目标攻击，而无目标攻击的有效性是根据有目标攻击结果来评估的。也就是说，如果一个图像可以被扰动到总共9个类中的至少一个目标类，则对该图像的非目标攻击成功。总的来说，它创造了总共36000个对抗性图像。

### B. ImageNet

 &emsp;&emsp; 在ImageNet上，我们采用无目标攻击，其DE的参数与CIFAR-10的一样。不同的是，在DE的过程中，我们的fitness函数是为了减少真实类别的概率。

 &emsp;&emsp; 我们的实验模型是BVLC AlexNet网络，使用了ILSVRC 2012测试集中的105个图片。还把这些图片从有损jpeg格式转换成了png格式。采用了center cropping预处理图片，将其resize成227x227大小。

### C. Results

 &emsp;&emsp; 在CIFAR-10和BVLC网络上的one-pixel攻击的成功率和置信度如下：

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/2.png?raw=true)

 &emsp;&emsp; 在Kaggle CIFAR-10上的three-pixel 和five-pixel attack的结果如下：

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/3.png?raw=true)

 &emsp;&emsp; 目标类别以及original-target的实验结果如下：

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/4.png?raw=true)

**Success Rate and Adversarial Probability Labels (Targeted Attack Results)**

 &emsp;&emsp; 在Kaggle CIFAR-10上的不同模型的成功率可以看出，one-pixel在不同模型之间具有泛化性，平均每个图片可以被扰动为两种目标类别。另外，通过增加扰动像素的数量，可攻击的目标类别的数量也增加了许多。

 &emsp;&emsp; 在ImageNet上，结果显示，one-pixel攻击对大尺寸的图像也泛化的很好。有22.91%的概率可以让ImageNet的16.04%的图像被误分类。

**Number ofTarget Classes (Non-targeted Attack Results)**

 &emsp;&emsp; 关于图5中所示的结果，我们发现只要一个像素的修改，就可以将相当数量的自然图像扰动到两个、三个和四个目标类。通过增加修改的像素数，对更多目标类的扰动变得非常可能。在非目标单像素攻击的情况下，VGG16网络对所提出的攻击具有略高的鲁棒性。这表明所有三种类型的网络（AllConv网络、NiN和VGG16）都易受此类攻击。

**Original-Target Class Pairs**

 &emsp;&emsp; 实验表明，有一些数据对更容易被扰动，比如类别3更容易被扰动成类别5.而且heat-map矩阵大部分是对称的

**Time complexity and average distortion**

 &emsp;&emsp; 我们使用进化的次数来评估时间复杂度，其中进化的次数等于候选解的个数乘以迭代次数。同时统计了单像素平均扰动大小。

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/5.png?raw=true)

**Comparing with Random One-Pixel Attack**

 &emsp;&emsp; 我们将我们的方法与随机的one-pixel攻击进行对比，来证明DE是有用的。我们对图像进行随机的选择，重复100次，每次随机的修改一个像素，采用随机的RGB值。实验发现，随机攻击的最高置信度等于DE方法的。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/6.png?raw=true)

**Change in fitness values**

 &emsp;&emsp; 经过实验发现，fitness的总体趋势是下降的。我们的目的也就是最小化fitness值。

## 5. Results On Original CIFAR-10 Test Data

 &emsp;&emsp; CIFAR-10的数据相比于Kaggle CIFAR-10的数据，具有更少的实际的噪音(旋转等)。与Kaggle上的模型相比，在NIN网络，移除了第二个average pooling层，对于全卷积网络，移除了第一层的batch normalization。其他参数一样

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89One%20Pixel%20Attack/7.png?raw=true)

## 6. Discussion

### A. Adversarial Perturbation

 &emsp;&emsp; 这篇文章证明了，沿着几个维度移动数据点就能够使其标签发生变化。同时，我们的结果证明了Goodfellow等人的假设：在一些维度上的小的扰动会累加，造成输出的大的改变。

 &emsp;&emsp; 实验结果表明，one pixel attack可以在不同的网络结构和不同的图像尺寸上进行泛化。通过更多的迭代或者更多的初始候选解集，扰动成功率应该可以进一步提高。同时，使用一个更好的进化算法，比如Co-variance Matrix Adaptation Evolution Strategy 也可以得到同样的效果。

### B. Robustness Of One-pixel Attack

 &emsp;&emsp; One-pixel的鲁棒性并不是很强。

## 7. Future Work

 &emsp;&emsp; 通过使用更好的进化算法，可以改进这个方法，比如Adaptive DE、d Covariance matrix adaptation evolution strategy (CMA-ES)。

 &emsp;&emsp; 基于进化的机器学习允许模型具有很大的灵活性，在一个称为神经进化的进化机器学习领域，通过进化计算不仅可以学习网络的权值，还可以学习网络的拓扑结构。

 &emsp;&emsp; 最后，self-organizing和novelty-organizing分类器可以通过使用灵活的representations来适应环境的变化。例如：例如，它们可以适应形状变化的迷宫，也可以适应整个实验中变量范围变化的问题（这是一个非常具有挑战性的场景，其中大多数甚至所有的深度学习算法都会失败）。这些显示，这是一条很有希望的道路，可以在未来纪念解决深层神经网络中的当前问题。