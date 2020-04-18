---
title: 机器学习（十二）PCA
date: 2020-04-03 14:12:05
tags:
 - [机器学习]
 - [PCA]
categories: 
 - [机器学习]
keyword: "机器学习,PCA"
description: "机器学习（十二）PCA"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89PCA/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、模型介绍

 &emsp;&emsp;  PCA(Principal Component Analysis)，即**主成分分析**方法，是一种使用最广泛的**数据降维**算法。

 &emsp;&emsp;  降维就是一种对高维度特征数据预处理方法。**降维是将高维度的数据保留下最重要的一些特征，去除噪声和不重要的特征，从而实现提升数据处理速度的目的**。在实际的生产和应用中，降维在一定的信息损失范围内，可以为我们节省大量的时间和成本。降维也成为应用非常广泛的数据预处理方法。降维的算法有很多，比如奇异值分解(SVD)、主成分分析(PCA)、因子分析(FA)、独立成分分析(ICA)。

 &emsp;&emsp;  PCA的主要思想是将**n维特征映射到k维上**，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。

 &emsp;&emsp;  PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中**方差最大的方向**，第二个新坐标轴选取是与第一个坐标轴正交的平面中使得方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面d个坐标轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面d个含有绝大部分方差的坐标轴。

<br>

## 1、算法思想

 &emsp;&emsp;  PCA是基于**最近重构性**和**最大可分性**而得到的方法。

- 最近重构性：**样本点到这个超平面的距离都足够近**
- 最大可分性：**样本点在这个超平面上的投影能尽可能分开**

 &emsp;&emsp;  PCA通过协方差矩阵的特征值分解来找到这些坐标轴（特征向量），其算法描述为：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89PCA/1.png?raw=true)

<br>

## 2、PCA推导

 &emsp;&emsp;  该过程是证明PCA的投影策略符合最近重构性和最大可分性。

**数据中心化**

 &emsp;&emsp;  首先，我们先对我们的数据中心化。
$$
x_i = x_i - \frac{1}{m}\sum_{i=1}^mx_i
$$
**最近重构性**

 &emsp;&emsp;  我们假定我们的数据（d维特征）经过了一个投影变换，投影变换后，得到的新坐标系为$\ \{w_1,w_2,...,w_d\}$  ，其中$\ w_i$ 是标准正交基向量，$\ ||w_i||_2=1$ 且$\ w_i^Tw_j=0(i\ne j)$ 。

 &emsp;&emsp;  我们可以丢弃新坐标系中的部分坐标，即将维度降低到$\ d'<d$ 。那么样本点$\ x_i$ 在低维空间坐标系中的投影为$\ z_i=(z_{i1};z_{i2};...;z_{id'})$ ，其中$\ z_{ij}=w_j^Tx_i$ 是$\ x_i$ 在低维坐标系下第$\ j$ 维的坐标，若基于$\ z_i$ 来重构$\ x_i$ ，则会得到$\ \hat{x_i} = \sum_{j=1}^{d'}z_{ij}w_j$ 。

 &emsp;&emsp;  考虑整个训练集，原样本点$\ x_i$ 与基于投影重构的样本点$\ \hat{x_i}$ 的距离为：
$$
\begin{equation}
\begin{split}
\sum_{i=1}^m||\sum_{j=1}^{d'}z_{ij}w_j - x_i||_2^2 &=\sum_{i=1}^m[(\sum_{j=1}^{d'}z_{ij}w_j)^2 - 2\sum_{j=1}^{d'}z_{ij}w_jx_i + x_i^2]\\
&=\sum_{i=1}^m[z_i^Tz_i - 2z_i^TW^Tx_i + x_i^2]\\
&=\sum_{i=1}^mz_i^Tz_i - 2\sum_{i=1}^mz_i^TW^Tx_i + const\\
&\propto -tr(W^T(\sum_{i=1}^mx_ix_i^T)W)\\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  其中$\ W = (w_1,w_2,...,w_{d'})$ 。

 &emsp;&emsp;  根据最近重构性，上面的式子应该被最小化，我们知道$\ w_j$ 是标准正交基，且$\ \sum_ix_ix_i^T$ 是协方差矩阵，所以我们的**目标函数**为：
$$
\begin{equation}
\begin{split}
min_W& \quad \  -tr(W^TXX^TW)\\
s.t.& \quad \  W^TW=I \\

\end{split}
\end{equation}
$$
**最大可分性**

 &emsp;&emsp;  我们知道，样本点$\ x_i$ 在新空间中超平面上的投影是$\ W^Tx_i$ ，若所有样本点的投影能尽可能分开，则应该使投影后样本点的方差最大化，而投影点后的方差是$\ \sum_iW^Tx_ix_i^TW$ ，于是基于最大可分性，我们得到的目标函数依然是：
$$
\begin{equation}
\begin{split}
min_W& \quad \  -tr(W^TXX^TW)\\
s.t.& \quad \  W^TW=I \\

\end{split}
\end{equation}
$$
**特征向量求解**

 &emsp;&emsp;  我们对上面的目标函数使用拉格朗日乘子法可得：
$$
XX^TW = \lambda W
$$
 &emsp;&emsp;  于是，我们只需要对协方差矩阵$\ XX^T$ 进行特征值分解，讲求得的特征值排序:$\ \lambda_1 \ge \lambda_2 \ge  ...\lambda_d $ ，再取前$\ d'$ 个特征值对应的特征向量构成$\ W=(w_1,w_2,...,w_{d'})$ ，这就是主成分分析的解。



**d‘的选取**

 &emsp;&emsp;  降维后低维空间的维数$\ d'$ 通常是由用户事先制定，我们可以设置一个阈值$\ t=95%$ ，使得$\ d'$ 满足：
$$
\frac{\sum_{i=1}^{d'}\lambda_i}{\sum_{i=1}^{d}\lambda_i} \ge t
$$
<br>

# 二、模型分析

## 1、模型优缺点

- **降维的优点**：

  - 使得数据集更易使用。
  - 降低算法的计算开销。

  - 去除噪声。

  - 使得结果容易理解。

- **优点**：
  - 仅仅需要以方差衡量信息量，不受数据集以外的因素影响。　
  - 各主成分之间正交，可消除原始数据成分间的相互影响的因素。
  - 计算方法简单，主要运算是特征值分解，易于实现。
- **缺点：**
  - 主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。
  - 方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。

<br>

## 2、核化线性降维

 &emsp;&emsp;  线性降维方法假设从高维空间到低维空间的哈数映射是线性的。然而，在不少现实任务中，可能需要非线性映射才能找到恰当的低维嵌入。如下图所示：经过PCA后，其丧失了其低维空间的结构。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%BA%8C%EF%BC%89PCA/2.png?raw=true)

 &emsp;&emsp;  非线性降维的一种常用方法，是**基于核技巧对线性降维方法进行“核化”**。

 &emsp;&emsp;  PCA是对下面的函数进行求解：
$$
(\sum_{i=1}^m z_iz_i^T)W = \lambda W
$$
 &emsp;&emsp;  易知：
$$
\begin{equation}
\begin{split}
W &= \frac{1}{\lambda}(\sum_{i=1}^m z_iz_i^T)W = \sum_{i=1}^m z_i \frac{z_i^TW}{\lambda} \\
&= \sum_{i=1}^m z_i \alpha_i\\

\end{split}
\end{equation}
$$
 &emsp;&emsp;  其中$\ \alpha_i = \frac{1}{\lambda}z_i^TW$ 。

 &emsp;&emsp;  假定$\ z_i$ 是由原始属性空间中的样本点$\ x_i$ 通过映射$\ \phi(x)$ 产生，即$\ z_i = \phi(x_i),i=1,2..m$ 。将其带入原PCA的目标函数中，我们可以得到：
$$
(\sum_{i=1}^m \phi(x_i)\phi(x_i^T))W = \lambda W
$$
 &emsp;&emsp;  且$\ W = \sum_{i=1}^m\phi(x_i)\alpha_i$。

 &emsp;&emsp;  我们引入核函数：
$$
k(x_i,x_j) = \phi(x_i)^T\phi(x_j)
$$
 &emsp;&emsp;  那么PCA的目标函数简化为：
$$
KA = \lambda A
$$
 &emsp;&emsp;  其中$\ K$ 为$\ k$ 对应的核矩阵，$\ (K)_{ij} = k(x_i,x_j)$ 。$\ A = (\alpha_1;\alpha_2;...;\alpha_m)$ 。

 &emsp;&emsp;  所以，**经过核化后，我们取上式的前$\ d'$ 个特征值对应的特征向量即可。**

 &emsp;&emsp;  对于新样本$\ x$ ，其投影后的第$\ j$ 维坐标维：
$$
\begin{equation}
\begin{split}
z_j &= w_j^T\phi(x) = \sum_{i=1}^m\alpha_i^j\phi(x_i)^T\phi(x) \\
&= \sum_{i=1}^m\alpha_i^jk(x_i,x)

\end{split}
\end{equation}
$$
 &emsp;&emsp;  其中$\ \alpha_i$ 已经过规范化，$\ \alpha_i^j$ 是$\ \alpha_i$ 的第$\ j$ 个分量。

 &emsp;&emsp;  根据上式，我们可以知道，为了获得投影后的坐标，核化的PCA需要对所有样本求和，因此它的计算开销较大。





**参考链接**

- 周志华老师的《机器学习》
- [主成分分析（PCA）原理详解](https://zhuanlan.zhihu.com/p/37777074)