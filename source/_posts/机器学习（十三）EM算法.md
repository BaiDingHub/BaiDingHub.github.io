---
title: 机器学习（十三）EM算法
date: 2020-04-03 14:13:05
tags:
 - [机器学习]
 - [EM算法]
categories: 
 - [机器学习]
keyword: "机器学习,PCA"
description: "机器学习（十二）PCA"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%B8%89%EF%BC%89EM%E7%AE%97%E6%B3%95/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、模型介绍

 &emsp;&emsp;  EM（Expectation-Maximum）算法也称**期望最大化算法**，曾入选“数据挖掘十大算法”中。EM算法是最常见的**隐变量估计方法**，在机器学习中有极为广泛的用途，例如常被用来学习高斯混合模型（Gaussian mixture model，简称GMM）、贝叶斯模型的参数；隐式马尔科夫算法（HMM）、LDA主题模型的变分推断等等。

 &emsp;&emsp;  EM算法是一种**迭代优化策略**，由于它的计算方法中每一次迭代都分两步，其中一个为期望步（E步），另一个为极大步（M步），所以算法被称为EM算法（Expectation-Maximization Algorithm）。EM算法受到**缺失思想**影响，最初是为了**解决数据缺失情况下的参数估计问题**。

 &emsp;&emsp;  其基本思想是：首先根据己经给出的观测数据，估计出模型参数的值；然后再依据上一步估计出的参数值估计缺失数据的值，再根据估计出的缺失数据加上之前己经观测到的数据重新再对参数值进行估计，然后反复迭代，直至最后收敛，迭代结束。

<br>

## 1、算法思想

 &emsp;&emsp;  未观察变量的学名是"**隐变量**"，令$\ X$ 表示已观测变量集，$\ Z$ 表示隐变量集，$\ \Theta$ 表示模型参数。若欲对$\ \Theta$ 做极大似然估计，则应最大化对数似然:
$$
LL(\Theta|X,Z) = ln\ P(X,Z|\Theta)
$$
 &emsp;&emsp;  然而由于$\ Z$ 是隐变量，上式无法直接求解。此时我们可通过对$\ Z$ 计算期望，来最大化已观测数据的对数"边际似然"：
$$
LL(\Theta|X) = ln\ P(X|\Theta) = ln\sum_{Z}P(X,Z|\Theta)
$$
 &emsp;&emsp;  EM算法是常用的估计参数因变量的利器，它是一种迭代式的算法，其基本思想是：若参数$\ \Theta$ 已知，则可根据训练数据推断出最优隐变量$\ Z$ 的值(E步)；反之，若$\ Z$ 的值已知，则可方便地对参数$\ \Theta$ 做极大似然估计(M步)。

 &emsp;&emsp;  于是，以初始值$\ \Theta^0$ 为起点，对于上面的极大似然估计，我们可以迭代的执行以下步骤直至收敛：

- 基于$\ \Theta^t$ 推断隐变量$\ Z$ 的期望，记为$\ Z^t$ ;
- 基于已观测变量$\ X$ 和$\ Z^t$ 对参数$\ \Theta$ 做极大似然估计，记为$\ \Theta^{t+1}$ 

<br>

 &emsp;&emsp;  进一步，若我们不是取$\ Z$ 的期望，而是基于$\ \Theta^t$ 计算隐变量$\ Z$ 的概率分布$\ P(Z|X,\Theta^t)$ ，则EM算法的两个步骤是：

- E步：以当前参数$\ \Theta^t$ 推断隐变量分布$\ P(Z|X,\Theta^t)$ ，并计算对数似然$\ LL(\Theta|X,Z)$ 关于$\ Z$ 的期望：

$$
Q(\Theta|\Theta^t) = \mathbb{E}_{Z|X,\Theta^t}LL(\Theta|X,Z)
$$

- M步：寻找参数最大化期望似然，即：

$$
\Theta^{t+1} = arg max_{\Theta} Q(\Theta|\Theta^t)
$$



## 2、算法示例---三硬币模型

在以下的示例中观测数据记为$\ Y$ (因为两个例子都是输出是观测数据)，隐藏变量(未观测变量)记为$\ z$，模型参数记为$\ \theta$ 。

**问题**

 &emsp;&emsp;  假设有三枚硬币A、B、C，每个硬币正面出现的概率是π、p、q。进行如下的掷硬币实验：先掷硬币A，正面向上选B，反面选C；然后掷选择的硬币，正面记1，反面记0。独立的进行10次实验，结果如下：1，1，0，1，0，0，1，0，1，1。假设只能观察最终的结果(0 or 1)，而不能观测掷硬币的过程(不知道选的是B or C)，问如何估计三硬币的正面出现的概率？

**求解**

 &emsp;&emsp;  在该示例中，我们并不知道每次实验时掷的时B还是C，这些便是隐变量$\ z$。我们需要在这些隐变量未知的情况下，求解$\ \pi,p,q$ 。

 &emsp;&emsp;  首先针对某个输出y值，它在参数$\ \theta(\theta=\pi,p,q)$下的**概率分布**为:
$$
P(y|\theta) = \sum_zP(y,z|\theta) = \sum_zP(z|\theta)P(y|z,\theta) = \pi p^y(1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}
$$
 &emsp;&emsp;  因此，针对观测数据$\ Y=(y_1,y_2,...,y_n)^T$ 的**似然函数**为：
$$
P(Y|\theta) = \sum_zP(Y,z|\theta) = \sum_zP(z|\theta)P(Y|z,\theta) = \prod_{j=1}^n\pi p_j^{y_j}(1-p)^{1-y_j} + (1-\pi) q_j^{y_j}(1-q)^{1-y_j}
$$
 &emsp;&emsp;  本题的目标就是求解参数$\ \theta$ ，使得上式最大。如果没有那些隐变量，我们完全可以通过极大似然估计的方法来求解参数，让对数似然求导等于0即可。但是隐变量的存在使得通过极大似然估计的方法及其复杂。下面根据EM算法来进行求解。

 &emsp;&emsp;  **E步**：在这一步，需要计算为观测数据的条件概率分布，也就是每一个$\ P(z|y_j,\theta)$ ，也就是我们需要知道每一步中掷B和掷C的概率。记$\ \mu_j$ 是在已知模型参数$\ \theta^i$ 下观测数据$\ y_j$ 来自掷硬币$\ B$ 的概率，相应的来自掷C的概率就是$\ 1-\mu_j$ 。
$$
P(z=掷B|y_j,\theta) = \mu_j = \frac{P(z=掷B，y_j|\theta^i)}{P(y_j|\theta^i)}=\frac{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j} }{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j}+(1-\pi^i)(q^i)^{y_j}(1-q^i)^{1-y_j} }
$$
 &emsp;&emsp;  **M步**：针对Q函数求导，Q函数的表达式为：
$$
\begin{equation}
\begin{split}
Q(\theta,\theta^i) &= \mathbb{E}_{Z|Y,\Theta^i}LL(\theta|Y,Z)\\
&= \sum_{j=1}^N\sum_zP(z|y_j,\theta^i)log\ P(y_j,z|\theta^i)\\
&=\sum_{j=1}^N \mu_j log[\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j}]+(1-\mu_j)log[(1-\pi^i)(q^i)^{y_j}(1-q^i)^{1-y_j}]\\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  展开上面的式子，同时求导，得到$\ \theta$ ：
$$
\begin{equation}
\begin{split}
\frac{\partial Q}{\partial \pi} &= (\frac{\mu_1}{\pi} - \frac{1-\mu_1}{1-\pi}) +...+(\frac{\mu_N}{\pi} - \frac{1-\mu_N}{1-\pi})\\
&=\frac{\mu_1 - \pi}{\pi(1-\pi)}+...+\frac{\mu_N - \pi}{\pi(1-\pi)}
\end{split}
\end{equation}
$$
 &emsp;&emsp;  令上式等于0，我们就可以得到$\ \pi^{i+1} = \frac{1}{N}\sum_{j=1}^N \mu_j$



# 二、模型介绍

## 1、模型应用

 &emsp;&emsp;  EM算法是一种求解隐含参数的一种方法，上面的示例是EM算法在**极大似然估计**或者说**贝叶斯理论**情况下的应用。

 &emsp;&emsp;  **K-means算法**也是EM算法的思路，在E步，初始化K个质心；在M步，计算每个样本到质心的距离，并把样本聚类到最近的之心内。不断地重复E步、M步直到之心不再变化为止。

 &emsp;&emsp;  EM算法还可以用在高斯混合模型（Gaussian mixture model，简称GMM）、隐式马尔科夫算法（HMM）等。

## 2、算法优缺点

- **缺点**
  - 所要优化的函数不是凸函数时，EM算法容易给出局部最佳解，而不是最优解。EM算法比K-means算法计算复杂，收敛也较慢，不适于大规模数据集和高维数据。



**参考链接**

- 周志华老师的《机器学习》
- [EM算法详解](https://zhuanlan.zhihu.com/p/40991784)
- [【机器学习算法系列之一】EM算法实例分析](https://chenrudan.github.io/blog/2015/12/02/emexample.html)

