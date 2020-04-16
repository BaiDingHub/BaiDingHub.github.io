---
title: 机器学习（十一）K-means
date: 2020-04-03 14:11:05
tags:
 - [机器学习]
 - [K-means]
categories: 
 - [机器学习]
keyword: "机器学习,K-means"
description: "机器学习（十一）K-means"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%B8%80%EF%BC%89K-means/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、模型介绍

 &emsp;&emsp;  k-means算法是一种**聚类算法**，所谓聚类，即根据相似性原则，将具有较高相似度的数据对象划分至同一类簇，将具有较高相异度的数据对象划分至不同类簇。聚类与分类最大的区别在于，聚类过程为**无监督过程**，即待处理数据对象没有任何先验知识，而分类过程为有监督过程，即存在有先验知识的训练数据集。

 &emsp;&emsp;  k-means算法中的**k代表类簇个数**，**means代表类簇内数据对象的均值**（这种均值是一种对类簇中心的描述），因此，k-means算法又称为k-均值算法。k-means算法是一种基于划分的聚类算法，以距离作为数据对象间相似性度量的标准，即数据对象间的距离越小，则它们的相似性越高，则它们越有可能在同一个类簇。数据对象间距离的计算有很多种，k-means算法通常采用欧氏距离来计算数据对象间的距离。



## 1、算法思想

 &emsp;&emsp;  K-means算法思想可描述为：首先初始化K个类簇中心$\ \mu_i$，，计算各个样本$\ x_j$ 到类簇中心$\ \mu_i$ 的距离，将样本分配给最近的那个类簇。处理完所有的样本后，我们得到了K个类簇的集合，更新这K个类簇的中心。然后继续计算距离，这样不断迭代，直到达到迭代次数或者类簇中心变化不大时迭代停止。伪代码如图：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%B8%80%EF%BC%89K-means/1.png?raw=true)

## 2、距离的选择

 &emsp;&emsp;  常用的距离包括欧氏距离、曼哈顿距离等，其他的距离可查看博主博客---机器学习之KNN



# 二、模型分析

## 1、模型效果

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%88%E5%8D%81%E4%B8%80%EF%BC%89K-means/2.png?raw=true)

## 2、模型优缺点

\- **优点：**
  算法简单易实现；
\- **缺点：**
  需要用户事先指定类簇个数K；
  聚类结果对初始类簇中心的选取较为敏感；
  容易陷入局部最优；
  只能发现球型类簇；



**参考链接**

- 周志华老师的《机器学习》
- [k-means算法详解](https://blog.csdn.net/zhihua_oba/article/details/73832614)