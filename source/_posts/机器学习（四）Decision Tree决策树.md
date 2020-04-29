---
title: 机器学习（四）Decision Tree决策树
date: 2020-04-03 14:04:05
tags:
 - [机器学习]
 - [Decision Tree]
categories: 
 - [机器学习]
keyword: "机器学习,Decision Tree"
description: "机器学习（四）Decision Tree决策树"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 1、模型介绍

 &emsp;&emsp; 机器学习中决策树是一个预测模型，它表示**对象属性和对象值之间的一种映射**，树中的每一个节点表示对象属性的判断条件，其分支表示符合节点条件的对象。树的叶子节点表示对象所属的预测结果。

 &emsp;&emsp; 根据数据的属性采用树状结构建立决策模型。决策树模型常常用来解决分类和回归问题。常见的算法包括 **CART** (Classification And Regression Tree)、**ID3**、**C4.5**、**随机森林 (Random Forest)** 等。



## 1.1 模型

 &emsp;&emsp;  一棵树包含一个根节点，若干个内部节点和若干个叶节点，叶节点对应决策结果，其他节点对应一个属性测试

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/1.png?raw=true)

## 1.2 决策树学习算法

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/3.png?raw=true)

决策树是一个递归过程，有三种情况导致递归返回：

- 当前样本包含的样本全属于同一类别，无需划分
- 当前样本属性集为空，或是所有样本在所有属性上取值相同，无法划分(取当前节点样本中类别最多的那一类作为分类)
- 当前节点包含的样本集合为空，不能划分(取父节点样本类别最多的作为分类)



## 1.3 决策树构建过程

### 1) 特征选择

 &emsp;&emsp; 特征选择是指在内部节点中选择一个特征来作为分类特征，特征选择决定了使用哪些特征来做判断。在训练数据集中，每个样本的属性可能有很多个，不同属性的作用有大有小。因而特征选择的作用就是筛选出跟分类结果相关性较高的特征，也就是分类能力较强的特征。

 &emsp;&emsp; 在特征选择中通常使用的准则是：**信息增益**、**增益率**、**基尼指数**

#### **信息增益**

 &emsp;&emsp; 信息熵(information entropy)是度量样本集合纯度的常用指标.

 &emsp;&emsp; 假定当前样本集合$\ D$ 中第$\ k$类样本所占的比例为 $\ p_k(k=1,2,…,|Y|)$,则$\ D$的信息熵为:
$$
Ent(D) = -\sum_{i=1}^{|Y|} p_k log_2(p_k)
$$
 &emsp;&emsp;  `Ent(D)`的值越小，D的纯度越高(约定：若$\ p=0$则$\ plog_2p=0$)

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/2.png?raw=true)

 &emsp;&emsp; 一般而言，信息增益越大，则意味着用属性a来进行划分所获得的纯度提升越大，因此，我们选择能够使信息增益最大的属性$\ a_*$ 作为该节点的分类特征：
$$
a_* = arg\ max_{a\in A} Gain(D,a)
$$
 &emsp;&emsp; `ID3`就是以信息增益为准则来选择划分属性的



#### **增益率**

 &emsp;&emsp; 实际上，信息增益对可取值数目较多的属性有所偏好(如编号，在西瓜集中若以编号为划分属性，则其信息增益最大)，为减少由于偏好而带来的不利影响，`C4.5`算法使用增益率(`gain ratio`)来选择最优划分属性:
$$
IV(a) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}\\
Gain\ ratio(D,a) = \frac{Gain(D,a)}{IV(a)}
$$
 &emsp;&emsp; `IV(a) `被称为称为属性a的固有值(intrinsic value),属性`a`的可能数目越多，则`IV(a)`的值通常越大。

 &emsp;&emsp; 然而，增益率准则对可取值数目较少的属性有所偏好，`C4.5`采用的是先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的



#### **基尼指数**

 &emsp;&emsp; 基尼指数是另一种数据的不纯度的度量方法，`CART`(Clasification and Regression Tree)使用基尼指数(Gini index)来选择划分属性

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/4.png?raw=true)

 &emsp;&emsp; 属性a的基尼指数定义为：
$$
Gini(D,a) = \sum_{v=1}^V\frac{|D^v|}{|D|}Gini(D^v)
$$
 &emsp;&emsp; 我们选择能够使基尼指数最大的属性$\ a_*$ 作为该节点的分类特征：
$$
a_* = arg\ max_{a\in A} Gini(D,a)
$$
注意：每次选择一个离散特征后，都要将该特征从特征集合中去除，即之后的节点不会再用该节点进行划分

### 2) 剪枝处理

 &emsp;&emsp; 剪枝(pruning)是决策树学习算法对付过拟合的主要手段，基本策略有预剪枝(prepruning)和后剪枝(post-pruning)

- 预剪枝：在决策树的生成过程中，对每个节点在划分前先进行估计，若当前节点的划分不能带来泛化性能提升则停止划分
- 后剪枝：先生成一个完整的树，然后自底向上对非叶节点考察，若将该节点对应的子数替换为叶节点能提升泛化性能则替换



#### **预剪枝**

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/5.png?raw=true)

 &emsp;&emsp; 预剪枝使决策树的很多分支都没有展开，不仅降低了过拟合的风险，还显著减少了训练时间和测试时间，但是可能会引起过拟合



#### 后剪枝

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/6.png?raw=true)

 &emsp;&emsp; 后剪枝通常比预剪枝保留更多的分值，一般情况下，后剪枝欠拟合风险很小，泛化性能优于预剪枝，但其训练时间比未剪枝和预剪枝都要大得多



### 3） 连续与缺失值处理

#### 连续值处理

 &emsp;&emsp; 前面讨论都是基于离散属性来生成决策树，对于连续属性可取数值不再有限，此时可以用连续属性离散化技术
 &emsp;&emsp; 最简单的策略:**二分法(bi-partition)**,这正是`C4.5`算法采用的机制

 &emsp;&emsp; 对连续属性a，我们可考察包含n-1个元素的候选划分点集合：
$$
T_a = \{\frac{a_i+a_{i+1}}{2},1\le i \le n-1\}
$$
 &emsp;&emsp; 即把区间$\ [a_i,a_{i+1})$ 的中位点作为候选划分点，然后就可像离散属性值一样来考察这些点：
$$
Gain(D,a) = max_{t \in T_a} Gain(D,a,t) \\
= max_{t \in T_a} (Ent(D) - \sum_{\lambda \in \{-,+\}} \frac{|D_t^{\lambda}|}{|D|}Ent(D_t))
$$
 &emsp;&emsp; 需要注意的是，与离散属性不同，若当前节点划分属性为连续属性，该属性仍可作为其后代节点的划分属性

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/7.png?raw=true)



#### 缺失值处理

- 如何在属性值缺失的情况下进行划分属性的选择？
- 给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分？

![8](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/8.png?raw=true)

![9](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/9.png?raw=true)





# 2、模型分析

## 2.1 模型优缺点

**优点**

- 决策树易于理解和解释，可以**可视化**分析，容易提取出规则；
- 可以同时处理标称型和数值型数据；
- 比较**适合处理有缺失属性**的样本；
- 能够处理不相关的特征；
- 测试数据集时，运行**速度比较快**；
- 在相对短的时间内能够对大型数据源做出可行且效果良好的结果。



**缺点**

- 容易发生**过拟合**（随机森林可以很大程度上减少过拟合）；
- 容易**忽略数据集中属性的相互关联**；
- 对于那些各类别样本数量不一致的数据，在决策树中，进行属性划分时，不同的判定准则会带来不同的属性选择倾向；信息增益准则对可取数目较多的属性有所偏好（典型代表ID3算法），而增益率准则（CART）则对可取数目较少的属性有所偏好，但CART进行属性划分时候不再简单地直接利用增益率尽心划分，而是采用一种启发式规则）（只要是使用了信息增益，都有这个缺点，如RF）。



## 2.2 多变量决策树

 &emsp;&emsp; 经过上面的分析我们可以发现，决策树在每次决策的时候只考虑了一个属性，其忽略了数据集属性的相互管理。

 &emsp;&emsp; 用专业的话讲就是，决策树所形成的分类边界有一个明显的特点：轴平行(axis-parallel),即它的分类边界由若干个与坐标轴平行的分段组成。这样的分类边界有较好的解释性，因为每段划分都直接对应了某个属性取值，但在分类任务比较复杂时，必须使用多段划分才能获得较好的近似。但若能使用斜的划分边界，决策树模型将大大简化。

![10](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/10.png?raw=true)

 &emsp;&emsp; **多变量决策树(multivariate decision tree)**就是能实现斜划分甚至更复杂划分的决策树(亦称斜决策树 oblique decision tree)
 &emsp;&emsp; 在此类决策树中，非叶节点不再是仅针对某个属性，而是针对属性的线性组合进行测试，每个非叶节点是一个形如$\ \sum_{i=1}^d w_ia_i=t$ 的线性分类器，$\ w_i$  和 $\ t$ 可在该结点所含的样本集和属性集上学得，它不是为每个非叶节点寻找一个最优划分属性，而是试图建立一个合适的线性分类器，如图：

![11](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%9B%9B%EF%BC%89Decision%20Tree%E5%86%B3%E7%AD%96%E6%A0%91/11.png?raw=true)



**参考链接**

- [决策树算法介绍及应用](https://www.ibm.com/developerworks/cn/analytics/library/ba-1507-decisiontree-algorithm/index.html)
- [西瓜书笔记——第四章 决策树](https://lovelyfrog.github.io/2018/03/06/第四章决策树/)