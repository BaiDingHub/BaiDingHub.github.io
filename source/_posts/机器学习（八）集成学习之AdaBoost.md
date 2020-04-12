---
title: 机器学习（八）集成学习之AdaBoost
date: 2020-04-03 14:08:05
tags:
 - [机器学习]
 - [集成学习]
categories: 
 - [机器学习]
keyword: "机器学习,集成学习"
description: "机器学习（八）集成学习之AdaBoost"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%85%AB%EF%BC%89%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8BAdaBoost/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、模型介绍

 &emsp;&emsp;   **AdaBoost方法的自适应在于：前一个分类器分错的样本会被用来训练下一个分类器**。AdaBoost方法是一种迭代算法，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率。**每一个训练样本都被赋予一个权重，表明它被某个分类器选入训练集的概率**。如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它被选中的概率就被降低；相反，如果某个样本点没有被准确地分类，那么它的权重就得到提高。通过这样的方式，AdaBoost方法能“聚焦于”那些较难分（更富信息）的样本上。虽然AdaBoost方法**对于噪声数据和异常数据很敏感**。但相对于大多数其它学习算法而言，却又不会很容易出现过拟合现象。

 &emsp;&emsp;  AdaBoost是一种"加性模型"，即基学习器的线性组合：
$$
H(x) = \sum_{t=1}^T \alpha_th_t(x)
$$
 &emsp;&emsp;  整体要做的事情，就是最小化指数损失函数：
$$
l_{exp}(H|D) = \mathbb{E}_{x \thicksim D}[e^{-f(x)H(x)}]
$$



## 1、训练过程

 &emsp;&emsp;  最初，训练集会有一个初始的样本权值分布（平均分布），基于训练集$\ D$ 与该分布$\ D_t$ 训练一个基学习器$\ h_t$ ，估计$\ h_t$ 的分类误差，根据误差确定分类器$\ h_t$的权重，根据误差等信息重新确定样本的权值分布$\ D_{t+1}$ （为分类错误的样本增加权重），伪代码如下图所示：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%85%AB%EF%BC%89%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8BAdaBoost/1.png?raw=true)

 &emsp;&emsp;  接下来的部分是伪代码中的公式推导，如果不想看的，可直接跳过这些，直接到**样本分布&重采样**后再看。

<br>

## 2、损失函数的确定

 &emsp;&emsp;  我们为什么要用指数损失函数呢？下面来看，当我们最小化指数损失函数是，我们是否能够得到最优的分类器$\ H(x)$ 。

 &emsp;&emsp;  计算损失函数对$\ H(x)$ 的偏导：
$$
\frac{\partial\ l_{exp}(H|D)}{\partial\ H(x)} = -f(x)e^{-f(x)H(x)} = -e^{-H(x)}P(f(x)=1|x) + e^{H(x)}P(f(x)=-1|x)
$$
 &emsp;&emsp;  令偏导数为0，我们可得：
$$
H(x) = \frac{1}{2}ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)}
$$
 &emsp;&emsp;  看一下，我们得到的$\ H(x)$ 是都是最优的，是否满足分类要求?
$$
\begin{equation}
\begin{split}
sign(H(x))&=sign(\frac{1}{2}ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x))})\\
&=
\begin{cases}
1 & & P(f(x)=1|x)>P(f(x)=-1|x)\\
-1 & & P(f(x)=1|x)<P(f(x)=-1|x)
\end{cases} \\
&=argmax_{y\in\{-1,1\}} P(f(x)=y|x)
\end{split}
\end{equation}
$$
 &emsp;&emsp;  我们可以知道，$\ sign(H(x))$ 达到了贝叶斯最优错误率，也就是说该损失函数是可以的。

<br>

## 3、**分类器权重$\ \alpha_t$ 的确定**

 &emsp;&emsp;  我们基于分布$\ D_t$ 训练出来了一个分类器$\ h_t$ ，那么我们该给他们分配怎样的权重，才能最小化损失函数呢？

 &emsp;&emsp; 观察$\ \alpha_th_t$ 的损失函数形式：
$$
\begin{equation}
\begin{split}
l_{exp}(\alpha_th_t|D_t)&=\mathbb{E}_{x\thicksim D_t}[e^{-f(x)\alpha_th_t(x)}]\\
&=\mathbb{E}_{x\thicksim D_t}[e^{-\alpha_t}Ⅱ(f(x)=h_t(x))+e^{\alpha_t}Ⅱ(f(x)\ne h_t(x))]\\
&=e^{-\alpha_t}P_{x\sim D_t}(f(x)=h_t(x)) + e^{\alpha_t}P_{x\sim D_t}(f(x)\ne h_t(x))\\
&=e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t}\epsilon_t
\end{split}
\end{equation}
$$
 &emsp;&emsp;  其中$\ \epsilon_t = P_{x\sim D_t}(h_t(x)\ne f(x))$ ，$\ Ⅱ(·)$ 表示概率，计算损失函数对$\ \alpha_t$ 的导数：
$$
\frac{\partial\ l_exp(\alpha_th_t|D_t)}{\partial\ \alpha_t} = -e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t}\epsilon_t
$$
  &emsp;&emsp;  令导数为0，我们可得：
$$
\alpha_t = \frac{1}{2}ln\frac{1-\epsilon_t}{\epsilon_t}
$$
<br>

## 4、样本权重分布的确定

 &emsp;&emsp;  AdaBoost算法，在获得$\ H_{t-1}$ 之后，样本分布将进行调整，使下一轮的基学习器$\ h_t$ 能纠正$\ H_{t-1}$ 的一些错误，理想的$\ h_t$ 能纠正$\ H_{t-1}$ 的全部错误，即最小化：
$$
\begin{equation}
\begin{split}
l_{exp}(H_{t-1}+h_t|D)&=\mathbb{E}_{x\thicksim D}[e^{-f(x)(H_{t-1}(x)+h_t(x))}] \\
&=\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}e^{-f(x)h_t(x))}]
\end{split}
\end{equation}
$$
 &emsp;&emsp;  我们对$\ e^{-f(x)h_t(x)}$ 进行泰勒展开，并考虑到$\ f^2(x)=h_t^2(x)=1$ ，我们得到：
$$
\begin{equation}
\begin{split}
l_{exp}(H_{t-1}+h_t|D)&\backsimeq\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac{f^2(x)h_t^2(x)}{2})] \\
&=\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac{1}{2})]
\end{split}
\end{equation}
$$
 &emsp;&emsp;  因此，我们要得到的理想的基学习器是：
$$
\begin{equation}
\begin{split}
h_t(x) &= argmin_h\ l_{exp}(H_{t-1}+h|D) \\
&=argmin_h\ \mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac{1}{2})]
\end{split}
\end{equation}
$$
 &emsp;&emsp;  在第$\ t$ 次迭代时，$\ H_{t-1}(x)$ 是已知的，因此可以将其当作常数，则上式转换为：
$$
\begin{equation}
\begin{split}
h_t(x)&=argmax_h\ \mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}f(x)h_t(x)] \\
&=argmax_h\ \mathbb{E}_{x\thicksim D}[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}]}f(x)h_t(x)] 
\end{split}
\end{equation}
$$
 &emsp;&emsp;  我们可以令$\ D_t$ 表示为下面这个分布：
$$
D_t(x)=\frac{D(x)e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}]}
$$
 &emsp;&emsp;  根据数学期望的定义，这等价于：
$$
\begin{equation}
\begin{split}
h_t(x)&=argmax_h\ \mathbb{E}_{x\thicksim D}[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t-1}(x)}]}f(x)h_t(x)]  \\
&= argmax_h\ \mathbb{E}_{x\thicksim D_t}[f(x)h(x)]
\end{split}
\end{equation}
$$
 &emsp;&emsp;  因为$\ f(x),h(x) \in \{-1,1\}$ ，那么：
$$
f(x)h(x) = 1 - 2Ⅱ(f(x)\ne h(x))
$$
 &emsp;&emsp;  那么，我们得到的理想的学习器，就是：
$$
\begin{equation}
\begin{split}
h_t(x) &= argmin_h \ \mathbb{E}_{x\sim D_t}[Ⅱ(f(x)\ne h(x))] \\
&= argmin_h \epsilon_t
\end{split}
\end{equation}
$$
 &emsp;&emsp;  因此，在分布$\ D_t$ 下，我们的弱分类器就能够得到最优值，根据残差逼近的思想，对分布进行迭代式的逼近，得到：
$$
\begin{equation}
\begin{split}
D_{t+1}(x)&=\frac{D(x)e^{-f(x)H_{t}(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t}(x)}]}\\
&= \frac{D(x)e^{-f(x)H_{t-1}(x)}e^{-f(x)\alpha_t h_t(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t}(x)}]}\\
&= D_t(x)e^{-f(x)\alpha_t h_t(x)}\frac{\mathbb{E}_{x\thicksim D}e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\thicksim D}[e^{-f(x)H_{t}(x)}]}\\
\end{split}
\end{equation}
$$
<br>

## **5、样本分布&重采样**

 &emsp;&emsp;  Boosting算法要求基学习器能对特定的数据分布进行学习，这可通过“**重赋权法**”实施，即在训练过程的每一轮中，根据样本分布为每个训练样本重新赋予一个权重。

 &emsp;&emsp;  对无法接受带权样本的基学习器算法，则可通过“**重采样法**”进行处理，即在每一轮学习中，根据样本分布对训练集重新进行采样，再用重采样而得到的样本集对基学习器进行训练。

<br>

## 6、AdaBoost示例

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%85%AB%EF%BC%89%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B9%8BAdaBoost/2.png?raw=true)

我们可以将基分类器设为线性分类器，那么迭代过程中的分类效果图如上图所示。

<br>

## 7、AdaBoost 总结

- 简单，不用做特征筛选
- 不用担心overfitting！
- adaboost是一种有很高精度的分类器
- 只适用于二分类任务

<br>



**参考链接**

- [AdaBoost算法详解](https://blog.csdn.net/zhuangxiaobin/article/details/26075667)
- 周志华老师的《机器学习》