---
title: 机器学习（二）线性判别分析LDA
date: 2020-04-03 14:02:05
tags:
 - [机器学习]
 - [LDA]
categories: 
 - [机器学习]
keyword: "机器学习,LDA"
description: "机器学习（二）线性判别分析LDA"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E4%BA%8C%EF%BC%89%E7%BA%BF%E6%80%A7%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90LDA/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 1、模型介绍

 &emsp;&emsp; 不同于PCA方差最大化理论，LDA算法的思想是将数据投影到低维空间之后，使得**同一类数据尽可能的紧凑，不同类的数据尽可能分散**。LDA也是一种线性分类器，其不需要迭代式的进行训练，可以根据数据集，利用优化算法直接得到权重。LDA是一种多分类的线性分类器，是一种监督学习模型。

 &emsp;&emsp; LDA有如下两个假设:

 &emsp;&emsp;  &emsp;&emsp; (1) 原始数据根据**样本均值**进行分类。

 &emsp;&emsp;  &emsp;&emsp; (2) 不同类的数据拥有相同的协方差矩阵。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E4%BA%8C%EF%BC%89%E7%BA%BF%E6%80%A7%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90LDA/1.gif?raw=true)



## 1.1 模型

 &emsp;&emsp;  对于N分类问题，模型输入$\ x$ ，模型参数$\ w$ ，模型输出$\ h(x)$ (N维向量)，预测标签$\ y$
$$
h(x) = w^Tx + w_0 \\
y = argmax_k\ h(x)
$$


## 1.2 权重计算过程

我们先以二分类任务为例，进行数学公式的推导。$\ D$ 表示数据集，$\ D_i$ 表示类别$\ i$ 的数据。

**计算类别 i 数据的原始中心点**：
$$
\mu_i = \frac{1}{n_i}\sum_{x \in D_i} x
$$
**计算类别 i 数据投影后的中心点**：
$$
\tilde{\mu_i} = w^T·\mu_i
$$
其用来衡量投影后，不同类别间的距离



**得到每个数据投影后的点**：
$$
z = w^Tx
$$
**计算类别 i 数据投影后的方差（数据之间的分散程度） **
$$
\tilde{s_i}^2 = \sum_{z\in Z_i}(z - \tilde{\mu_i})^2
$$
其用来衡量投影后，同一类别内数据的距离



**最终，得到投影后的损失函数（二分类）**
$$
J(w) = \frac{(\tilde{\mu_1} - \tilde{\mu_2})^2}{\tilde{s_1}^2+\tilde{s_2}^2}
$$
对上面的式子进行带入展开
$$
J(w) = \frac{(w^T\mu_1 - w^T\mu_2)^2}{\sum_{x\in D_1}(w^Tx-w^T\mu_1)^2+\sum_{x\in D_2}(w^Tx-w^T\mu_2)^2} \\ \qquad
=\frac{w^T(\mu_1 - \mu_2)(\mu_1 - \mu_2)^Tw}{\sum_{x\in D_1}(w^T(x - \mu_1)(x - \mu_1)^Tw)+\sum_{x\in D_2}(w^T(x - \mu_2)(x - \mu_2)^Tw)}
$$
我们记$\ S_B=(\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$ ,$\ S_i = \sum_{x\in D_i}(x-\mu_i)(x-\mu_i)^2$ ，$\ S_w = \sum_i^NS_i$

则上面的式子，我们可以简化为：
$$
J(w) = \frac{w^TS_Bw}{w^TS_ww}
$$


**推广到多分类**

在多分类任务中，$\ S_B$ 等于所有的$\ \mu$ 两两相减的平方 的加和，$\ S_w$ 等于所有的$\ S_i$ 的加和



**使用拉格朗日乘子法求解权重**

在拉格朗日乘子法中，我们需要对分母进行限制，限制其等于1。这个限制条件跟SVM中的限制是一个道理，是为了防止权重的倍数增长而造成的有无穷多个解的问题，比如1/1=1,而2/2=1，我们加了限制条件，就只有了第一种可能性，这让我们可以使用拉格朗日算法
$$
c(w) = w^TS_Bw - \lambda(w^TS_ww-1)
\\
\Rightarrow \frac{dc}{dw} = 2S_Bw - 2\lambda S_ww = 0
\\
\Rightarrow  2S_Bw = 2\lambda S_ww
$$
我们可以利用线性代数求特征值的方法求解w






# 2、模型分析

## 2.1 LDA与PCA

 &emsp;&emsp; LDA是一种利用降维方法而得出的线性分类器，PCA是一种降维方法。我们可以根据降维这个话题，来分析这两个模型。LDA在降维时，主要利用了模型的均值，来进行类别的区分，而PCA则是使用了方差。如图：

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E4%BA%8C%EF%BC%89%E7%BA%BF%E6%80%A7%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90LDA/2.png?raw=true)



## 2.2 模型优缺点

**优点**

- 计算速度快
- 充分利用了先验知识



**缺点**

- 当数据不是高斯分布时候，效果不好，PCA也是。
-  降维之后的维数最多为 类别数-1，因为在进行拉格朗日求解时，我们能够得到的特征向量的数目为x的秩-1。



## 2.3 模型应用

- 可以用于**多分类任务**中
- 降维之后的维数最多为类别数-1。所以当数据维度很高，但是类别数少的时候，算法并不适用。



**参考链接**

- [机器学习-LDA(线性判别降维算法)](https://zhuanlan.zhihu.com/p/51769969)
- [机器学习中的数学(4)-线性判别分析（LDA）, 主成分分析(PCA)](https://www.cnblogs.com/leftnoteasy/archive/2011/01/08/lda-and-pca-machine-learning.html)