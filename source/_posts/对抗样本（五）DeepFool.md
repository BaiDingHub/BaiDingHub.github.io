---
title: 对抗样本（五）DeepFool
date: 2020-04-03 13:05:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,DeepFool"
description: "对抗样本（五）DeepFool"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E4%BA%94%EF%BC%89DeepFool/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **DeepFool: a simple and accurate method to fool deep neural networks**

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2015年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp;[https://arxiv.org/abs/1511.04599](https://arxiv.org/abs/1511.04599)



# 二、论文背景及简介

 &emsp; &emsp; 目前，没有有效率的方法可以用来精确的计算深度模型对对抗扰动的鲁棒性。在这篇论文中，提出了**DeepFool**的算法来生成扰动，并且**提出了一种量化分类器鲁棒性的方法**。

<br>

# 三、论文内容总结

- 提出了一种计算分类器对对抗扰动的鲁棒性的评价方法
- FGSM虽然快，但是它只是提供了最优扰动的一个粗略的估计，它执行的梯度的方法，经常得到的是局部最优解，DeepFool能够得到更小的扰动，甚至比FGSM小一个数量级
- 提出了一个新的对抗攻击方法DeepFool：

$$
r_*(x_0)=argmin||r||_2  \\ s.t. \ \ sign(f(x_0+r)) \ne sign(f(x_0))=-\frac{f(x_0)}{||w||_2^2}w
$$

- 在DeepFool中可以采用任意的lp norm
- DeepFool训练出来的对抗样本进行Fine-tuning后，网络的鲁棒性变的更好。FGSM的Fine-tuning却让网络的鲁棒性变差。作者认为：用变动过大的扰动来进行Fine-tuning会让网络的鲁棒性变差。但博主认为：大的扰动在Fine-tuning后之所以让鲁棒性变差，是因为实验所使用的Epoch太少了，不足以让网络能够清楚地学习到大的扰动所带来的影响，才让鲁棒性变差，而增加Epoch或者增加网络的复杂性，就可以训练的很好。这只是个人理解。

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://baidinghub.github.io/2020/04/03/对抗攻击之目录/)

<br>

# 四、论文主要内容

## 1、Introduction

 &emsp; &emsp; 针对样本x，标签$\ \hat{k}(x)$ ，我们可以用如下关系来生成对抗样本：
$$
\bigtriangleup(x,\hat{k}) = min_r||r||_2 \ \ s.t.\ \hat{k}(x+r) \neq \hat{k}(x)
$$
 &emsp; &emsp; 我们，可以把$\ \bigtriangleup(x,\hat{k})$ 称作$\ \hat{k}$ 在点$\ x$ 上的鲁棒性，分类器$\ \hat{k}$ 的鲁棒性可以如下定义：
$$
\rho_{adv}(\hat{k}) = E_x \frac{\bigtriangleup(x,\hat{k})}{||x||_2}
$$
 &emsp; &emsp; 其中$\ E_x$ 是对数据分布的期望。

 &emsp; &emsp; 对抗扰动的研究帮助我们明白了对一个分类器来说，什么样的特征被使用了。一个准确的寻找对抗扰动的方法对于研究和比较不同分类其对对抗样本的鲁棒性是十分必要的，这可能有助于更好的理解目前结构的限制，然后能够找到方法来增加鲁棒性。目前，还没有很好的方法被提出来得到对抗扰动，这篇论文就弥补了这一缺陷。

 &emsp; &emsp; 这篇论文主要的贡献如下：

- **提出了一个简单的精确的方法来计算对抗扰动以及比较不同分类器的鲁棒性**
- 进行了一次实验对比，发现由我们的方法得出来的对抗扰动 更可信 更有效率，而且，使用对抗样本来增强实验数据能够极大的增加对对抗扰动的鲁棒性
- 使用不精确的方法来计算对抗性扰动可能会导致关于鲁棒性的不同结论，有时甚至是误导性的结论。因此，我们的方法可以更好地理解这种有趣的现象及其影响因素

 &emsp; &emsp; 这篇论文提到，FGSM虽然快，但是它只是提供了最优扰动的一个粗略的估计，它执行的梯度的方法，经常得到的是局部最优解。

<br>

## 2、DeepFool For Binary Classifiers

 &emsp; &emsp; 在该节，$\ \hat{k}(x) =sign(f(x))$ ，$\ f(x)$ 是一个二值分类器，我们把 $\ \mathcal{F}=\{x:f(x)=0\}$ 记为分类器的边界。$\ f(x) = w^T x+b$  。

 &emsp; &emsp; 如果$\ f$ 是一个线性的，我们可以很容易的看出$\ f$ 在点$\ x_0$ 出的鲁棒性。$\ \bigtriangleup(x_0;f)$ 等于$\ x_0$ 到分类边界$\ \mathcal{F}$ 的距离，也可以用$\ r_*(x_0)$ 表示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182210610.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 其公式如下：
$$
r_*(x_0)=argmin||r||_2  \\ s.t. \ \ sign(f(x_0+r)) \ne sign(f(x_0))=-\frac{f(x_0)}{||w||_2^2}w
$$
 &emsp; &emsp; 如果$\ f$ 是一个一般的可微的二值分类器，我们将会采用一个迭代的操作来估计其鲁棒性。在每次迭代过程中，在点$\ x_i$ 的附近，$\ f$ 是线性的，线性分类器的扰动可以这样计算：
$$
argmin_{r_i} ||r_i||_2 \\
s.t. \ \ f(x_i) + \bigtriangledown f(x_i)^Tr_i=0 
$$
 &emsp; &emsp; 迭代过程为：先用上面线性的公式来得到$\ r_i$ ，然后更新$\ x_i=x_0+r_i$ ，进行不断迭代。当$\ f(x_{i+1})$ 改变了符号时，迭代停止。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182238878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 实际上，上述算法经常会收敛到$\ \mathcal{F}$ 边界上的一个点，为了到达分类边界的另一边，最后的扰动向量$\ \hat{r}$ 通常会乘以一个常数$\ 1+\eta$ ，在实验中，$\ \eta$ 取0.02。

<be>

## 3、DeepFool For Multiclass classifiers

 &emsp; &emsp; 在多分类分类器上的DeepFool实际上可以看作是，在二值分类器上DeepFool的聚合。在这里，假设多分类分类器由$\ c$ 类。我们可以知道 $\ \hat{k}(x) = argmax_kf_k(x)$ 

### 3.1 Affine Multiclass Classifier（线性的多分类分类器）

 &emsp; &emsp; 对于线性多分类的分类器来说，我们只需要：
$$
argmin_r||r||_2 \\
s.t. \exist k:w_k^T(x_0+r) + b_k \geq w_{\hat{k}(x_0)}^T(x_0+r) + b_{\hat{k}(x_0)}
$$
 &emsp; &emsp; $\ w_k$ 是$\ W$ 的第k列，即是类别k的，权重。

 &emsp; &emsp; 上面的问题，我们可以转化成，针对 $\ x_0$ 与 凸多面体P的补的运算，其中凸多面体的定义如下：
$$
P = \bigcap_{k=1}^c\{x:f_{\hat{k}(x_0)}(x)\geq f_k(x)\}
$$
 &emsp; &emsp; 这个凸多面体，表示的是可以正确分类的域，所以$\ x_0$ 是 在P内部的。而我们要想将其误分类，我们就要让$\ x_0$ 进入到P的补集的域中，于是我们需要用到$\ x_0 与P^c的距离，dist(x_0,P^c)$ ，我们记$\ x_0$ 离P的边界最近的标签为$\ \hat{l}(x_0)$ ，对于下面这个图来说，$\ \hat{l}(x_0)=3$ ：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182253731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
 &emsp; &emsp; 其实，我们可以用如下公式来计算$\ \hat{l}(x_0)$ ：
$$
\hat{l}(x_0) = argmin_{k\ne \hat{k}(x_0)} \frac{|f_k(x_0)-f_{\hat{k}(x_0)}(x_0)|}{||w_k - w_{\hat{k}(x_0)}||_2}
$$
 &emsp; &emsp; 在分母上加l2范数，只是为了得到的值免受权重的大小的影响，该公式的意思，其实就是两个$\ f(x_0) $ 之间的距离。

 &emsp; &emsp; 因此，我们可以将$\ x_0$ 误分类为$\ \hat{l}(x_0)$ ，扰动为：
$$
r_*(x_0) = \frac{|f_{\hat{l}(x_0)}(x_0)-f_{\hat{k}(x_0)}(x_0)|}{||w_{\hat{l}(x_0)} - w_{\hat{k}(x_0)}||_2^2} (w_{\hat{l}(x_0)} - w_{\hat{k}(x_0)})
$$
<br>

### 3.2 General Classfier

 &emsp; &emsp; 在本节，我们将拓展DeepFool算法到非线性的可微的多分类器中。

 &emsp; &emsp; 在非线性的情况下，我们依然使用迭代的方法，在每次迭代过程中，P可用如下表述：
$$
\tilde{P_i} =\bigcap_{k=1}^c\{x:f_k(x_i)-f_{\hat{k}(x_0)}(x_i) + \bigtriangledown f_k(x_i)^Tx - \bigtriangledown f_{\hat{k}(x_0)}(x_i)^Tx\le0\}
$$
 &emsp; &emsp; 其迭代方法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182309436.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 该迭代算法是贪婪算法，没有被证实一定可以收敛到最优解，但能够得到最小扰动的良好的近似。

 &emsp; &emsp; DeepFool与优化算法紧密相连，在二值分类器情形下，它可以看作是**牛顿迭代算法**，用于在欠定情形下求非线性方程组的根，这种算法被称为正规流法。这个算法，也可以被看成是梯度下降方法，在每次迭代时自动选择自适应步长。在多分类器的算法中的线性化也类似于**序列凸规划**，其中约束在每一步都被线性化。

<br>

### 3.3 Extension to lp norm

 &emsp; &emsp; 之前所使用的公式中都是使用的$\ l_2\ norm$ ，其实，我们也可以使用$\ l_p \ norm$ ，我们的算法也依旧可以找到比较好的结果。

 &emsp; &emsp; 相应的公式，更改成：
$$
\hat{l} = argmin_{k\ne\hat{k}(x_0)}\frac{f_k'}{||w_k'||_q} \\
r_i = \frac{f_{\hat{l}}'}{||w_{\hat{l}}'||_q^q} |w_{\hat{l}}'|^{q-1} · sign(w_{\hat{l}}')
$$
 &emsp; &emsp; 其中$\  q = \frac{p}{p-1}$

 &emsp; &emsp; 特别的，当$\ p=\infty$ 时，公式为：
$$
\hat{l} = argmin_{k\ne\hat{k}(x_0)}\frac{f_k'}{||w_k'||_1} \\
r_i = \frac{f_{\hat{l}}'}{||w_{\hat{l}}'||_1} · sign(w_{\hat{l}}')
$$
<br>

## 4、实验结果

### 4.1 准备工作

 &emsp; &emsp; 所使用的数据集，以及相应的网络如下：

- MNIST：一个两层的全连接网络，以及一个两层的LeNet卷积网络。这两个网络都是用SGD和Momentum来训练的。
- CIFAR-10：一个三层的LeNet网络，以及NIN网络
- ISCVRC 2012：CaffeNet以及GoogLeNet预训练模型

 &emsp; &emsp; 对对抗扰动的鲁棒性评估如下，使用的平均鲁棒性：
$$
\hat{\rho}_{adv}(f) = \frac{1}{|\mathcal{D}|} \sum_{x\in \mathcal{D}}\frac{||\hat{r}(x)||_2}{||x||_2}
$$
 &emsp; &emsp; $\ \mathcal{D}$ 表示测试集。

 &emsp; &emsp; 作者拿DeepFool与L-BFGS以及FGSM的方法进行对比。
$$
\hat{r}(x) = \epsilon sign(\bigtriangledown_xJ(\theta,x,y))
$$
<br>

### 4.2 结果

 &emsp; &emsp; 各个模型的测试的准确率、平均鲁棒性以及一个对抗样本生成的时间的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182410656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 从实验可以看出，**DeepFool能够得到更小的扰动，甚至比FGSM小一个数量级**。而且，**DeepFool花费的时间是比L-BFGS要少**的，因为L-BFGS的优化的目标函数代价太高，而DeepFool会在很少的迭代后就能够收敛。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182349673.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

<br>

### 4.3使用对抗样本Fine-tunning

 &emsp; &emsp; 作者使用对抗样本对上述的模型进行Fine-tuning，并用FGSM的对抗样本进行对比。

 &emsp; &emsp; 作者对每一个网络在扰动了的训练集上Fine-tuning了5个epoch，并且使用了50%的学习率下降。为了做对比，作者也对每个网络在原始训练集上进行了相同的操作。

 &emsp; &emsp; 在四个网络上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182425385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 微调后的网络，在面对攻击时的准确率如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182442870.png)

 &emsp; &emsp; 可以看到，使用由**DeepFool训练出来的对抗样本进行Fine-tuning后，网络的鲁棒性变的更好**了。令人吃惊的是，**FGSM的Fine-tuning却让网络的鲁棒性变差**了，作者认为，这是因为由FGSM生成的扰动太大了，**用变动过大的扰动来进行Fine-tuning会让网络的鲁棒性变差**。作者认为，FGSM起的作用更像是正则化，而不能够代替原始数据来进行训练。作者也为此进行了实验：用DeepFool，使用$\ \alpha=1,2,3$ ，生成对抗样本，来进行Fine-tuning，得到的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182452587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 因此，设计一个方法来得到最小的扰动，是很重要的。

 &emsp; &emsp; 博主在此时认为，大的扰动在Fine-tuning后之所以让鲁棒性变差，是因为实验所使用的Epoch太少了，不足以让网络能够清楚地学习到大的扰动所带来的影响，才让鲁棒性变差，而增加Epoch或者增加网络的复杂性，就可以训练的很好。这只是个人理解，还未证明。

 &emsp; &emsp; 而，神奇的是，在NIN网络上，作者此次采用了90%的学习率衰减，在FGSM和DeepFool上做实验，得到的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200319182502104.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 这次的实验结果，与上面的实验结果就不同了，而且，Fine-tuning一个Epoch不足以说明其影响。这也就说明，**使用一个精确的工具来测量分类器的鲁棒性对于得出关于网络鲁棒性的结论是至关重要的**

<br>

## 5、Conclusion

 &emsp; &emsp; 这篇论文，提出了一个算法，**DeepFool**。它采用了**迭代**的方法来生成对抗样本。并且进行了大量的实验证明了该方法在计算对抗扰动以及效率上的优势。同时，提出了一个方法来**评价分类器的鲁棒性**。
