---
title: AI小知识系列(五) 面试小知识(2)
date: 2020-04-03 11:15:05
tags:
 - [AI小知识]
 - [面试小知识]
categories: 
 - [深度学习,AI小知识]
keyword: "深度学习,AI小知识,面试小知识"
description: "AI小知识系列(五) 面试小知识(2)"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/AI%E5%B0%8F%E7%9F%A5%E8%AF%86/AI%E5%B0%8F%E7%9F%A5%E8%AF%86%E7%B3%BB%E5%88%97(%E4%BA%94)%20%20%E9%9D%A2%E8%AF%95%E5%B0%8F%E7%9F%A5%E8%AF%86(2)/cover.jpg?raw=true
---

<meta name="referrer" content="no-referrer"/>



# AI小知识系列--第二节

## 1、机器学习中样本不平衡的处理方法

 &emsp;&emsp;  处理样本不均衡的方法主要有以下三种：

### 1.欠采样

 &emsp;&emsp;  欠采样（undersampling）法是去除训练集内一些多数样本，使得两类数据量级接近，然后在正常进行学习。

 &emsp;&emsp;  这种方法的缺点是就是放弃了很多反例，这会导致平衡后的训练集小于初始训练集。而且如果采样随机丢弃反例，会损失已经收集的信息，往往还会丢失重要信息。

**欠采样改进方法1**

 &emsp;&emsp;  我们可以更改抽样方法来改进欠抽样方法，比如把多数样本分成核心样本和非核心样本，非核心样本为对预测目标较低概率达成的样本，可以考虑从非核心样本中删除而非随机欠抽样，这样保证了需要机器学习判断的核心样本数据不会丢失。

**欠采样改进方法2**

 &emsp;&emsp;  另外一种欠采样的改进方法是 EasyEnsemble 提出的继承学习制度，它将多数样本划分成若 N个集合，然后将划分过后的集合与少数样本组合，这样就形成了N个训练集合，而且每个训练结合都进行了欠采样，但从全局来看却没有信息丢失。

### 2.过采样

 &emsp;&emsp;  过采样（oversampling）是对训练集内的少数样本进行扩充，既增加少数样本使得两类数据数目接近，然后再进行学习。

 &emsp;&emsp;  简单粗暴的方法是复制少数样本，缺点是虽然引入了额外的训练数据，但没有给少数类样本增加任何新的信息，非常容易造成过拟合。

**过采样改进方法1**

 &emsp;&emsp;  使用数据增强方法

**过采样代表算法：SMOTE 算法**

 &emsp;&emsp;  SMOTE[Chawla et a., 2002]是通过对少数样本进行插值来获取新样本的。比如对于每个少数类样本a，从 a最邻近的样本中选取 样本b，然后在对 ab 中随机选择一点作为新样本。

### 3.阈值移动

 &emsp;&emsp;  这类方法的中心思想不是对样本集和做再平衡设置，而是对算法的决策过程进行改进。

 &emsp;&emsp;  当我们的训练集具有$\ m^+$ 个正例，$\ m^-$ 个反例时，我们可以修改分类阈值，即只有当$\ \frac{y}{1-y} > \frac{m^+}{m^-}$ 时，分类为正例

<br>

## 2、LR为什么使用交叉熵损失函数，其与最大似然函数是什么关系

 &emsp;&emsp;  LR通过sigmoid输出的值为样本分类为正例的概率，我们根据最大似然函数，可以得到每个样本分类正确的概率为$\ \hat{y_i}^{y_i}(1-\hat{y_i})^{1-y_i}$ ，给概率取Log（防止连乘造成的下溢），得到$\ y_ilog(\hat{y_i}) + (1-y_i)log(1-\hat{y_i})$ ，再取负数，就得到了我们的交叉熵损失函数。

<br>

## 3、为什么交叉熵损失可以提高具有sigmoid输出的模型的性能，而使用均方误差损失则会存在很多问题

 &emsp;&emsp;  我们知道Sigmoid有一个特点，即当输入很大或者很小时，其梯度接近0，这就使得我们在更新参数时更新的很慢。

 &emsp;&emsp;  我们以LR为例，输入为$\ x$，权重为$\ w$ ，经过线性分类器得到$\ z = wx+b$ ，经过Sigmoid函数$\ y_i = \sigma(z)$ 。我们看一下交叉熵损失函数和均方差损失函数对于模型的参数的梯度是怎么样的。

 &emsp;&emsp;  我们知道交叉熵损失函数为：
$$
L(y_i,\hat{y}_i) = -(y_ilog\ \hat{y_i} + (1-y_i)log (1-\hat{y_i}))
$$
 &emsp;&emsp;  均方差损失函数为：
$$
M(y_i,\hat{y_i}) = \frac{1}{2}(y-\hat{y_i})^2
$$
**分析均方差损失函数**

 &emsp;&emsp;  我们先来看一下均方差损失函数对于参数$\ w$ 的梯度情况：
$$
\begin{equation}
\begin{split}
\frac{\partial{M(y_i,\hat{y_i})} }{ {\partial \hat{y_i} } }&=\hat{y_i}-y_i \\
\frac{\partial{\hat{y_i} } }{ {\partial z} }&=\sigma'(z) \\
\frac{\partial{z} }{ {\partial w} }&=x\\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  于是我们可以得到：
$$
\frac{\partial{M(y_i,\hat{y_i})} }{ {\partial w}}=\frac{\partial{M(y_i,\hat{y_i})} }{ {\partial \hat{y_i} } }·\frac{\partial{\hat{y_i} } }{ {\partial z} }·\frac{\partial{z} }{ {\partial w} } = (\hat{y_i}-y_i)·\sigma'(z)·x
$$
 &emsp;&emsp;  也就是说，均方差损失函数对输入x的梯度于Sigmoid的梯度相似，同样包含了Sigmoid的梯度消失问题。

**分析交叉熵损失函数**

 &emsp;&emsp;  我们先来看一下交叉熵损失函数对于Sigmoid的输入x的梯度情况：
$$
\begin{equation}
\begin{split}
\frac{\partial{L(y_i,\hat{y_i})} }{ {\partial \hat{y_i} } }&=\frac{1-y_i}{1-\hat{y_i} }-\frac{y_i}{\hat{y_i} }\\
\frac{\partial{\hat{y_i}} }{ {\partial z} }&=\sigma'(z)\\
\frac{\partial{z} }{ {\partial w} }&=x\\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  于是我们可以得到：


$$
\begin{equation}
\begin{split}
\frac{\partial{L(y_i,\hat{y_i})} }{ {\partial w} }&=\frac{\partial{L(y_i,\hat{y_i})} }{ {\partial \hat{y_i} } }·\frac{\partial{\hat{y_i} } }{ {\partial z} }·\frac{\partial{z} }{ {\partial w} } = (\frac{1-y_i}{1-\hat{y_i} }-\frac{y_i}{\hat{y_i} })·\sigma'(z)·x \\
&=(\frac{1-y_i}{1-\sigma(z)}-\frac{y_i}{\sigma(z)})·\sigma'(z)·x \\
故\frac{\partial{L(y_i,\hat{y_i})} }{ {\partial w} }&=[(1-y_i)\sigma(z)-y_i(1-\sigma(z))]x \\
&=(\sigma(z)-y_i)x \\
&=(\hat{y}_i-y_i)x \\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  我们可以根据Sigmoid函数得到：
$$
\sigma'(z)= \frac{-e^{(-z)} }{(1+e^{(-z)})^2} = \sigma(z)(1-\sigma(z)) \\
$$
 &emsp;&emsp;  所以我们最后的结果为：
$$
\begin{equation}
\begin{split}
\frac{\partial{L(y_i,\hat{y_i})} }{ {\partial w} }&=[(1-y_i)\sigma(z)-y_i(1-\sigma(z))]x \\
&=(\sigma(z)-y_i)x \\
&=(\hat{y}_i-y_i)x \\
\end{split}
\end{equation}
$$
 &emsp;&emsp;  因此，交叉熵损失函数解决了Sigmoid造成的梯度消失问题，加快了参数更新的速度。

<br>

## 4、SVM 和 LR 的联系和区别

### 相同点

1. 都是监督的分类算法
2. 都是线性分类方法
3. 都是判别模型

### 不同点

#### 1. 两者的本质不同在于其Loss函数不同

 &emsp;&emsp;  LR的损失函数时**交叉熵损失函数**：
$$
L_{LR} = -\frac{1}{N}\sum_{i=1}^N(y_ilog\ \hat{y_i} + (1-y_i)log(1-\hat{y_i}))
$$
 &emsp;&emsp;  SVM的损失函数是**hinge Loss:**
$$
L_{SVM} = \sum_{i=1}^Nmax(0,1-w^Tx_iy_i)
$$
 &emsp;&emsp;  不同的loss function代表了不同的假设前提，也就代表了不同的分类原理。

 &emsp;&emsp;  LR方法基于**概率理论**，SVM基于几何间隔最大化原理。

 &emsp;&emsp;  **SVM只考虑分类面上的点，而LR考虑所有点**，**Linear SVM不直接依赖于数据分布**，分类平面不受异常点影响；LR则是受所有数据点的影响，所以受数据本身分布影响的。

#### 2. SVM依赖于数据的测度，而LR则不受影响

 &emsp;&emsp;  因为**SVM是基于距离的**，而**LR是基于概率的**，所以LR是不受数据不同维度测度不同的影响，而SVM因为要最小化$\ \frac{1}{2}||w||^2$ 所以其依赖于不同维度测度的不同，如果差别较大需要做normalization。如果不归一化，各维特征的跨度差距很大，目标函数就会是“扁”的，在进行梯度下降的时候，梯度的方向就会偏离最小值的方向，走很多弯路。

<br>

#### 3. SVM自带结构风险最小化，LR则是经验风险最小化

 &emsp;&emsp;  因为SVM本身就是优化$\ \frac{1}{2}||w||^2$ 最小化的，所以其优化的目标函数本身就含有结构风险最小化，所以不需要加正则项
 &emsp;&emsp;  而LR不加正则化的时候，其优化的目标是经验风险最小化，所以最后需要加入正则化，增强模型的泛化能力。

#### 4. SVM会用核函数而LR一般不用核函数

 &emsp;&emsp;  SVM转化为对偶问题后，分类只需要计算与少数几个支持向量的距离，这个在进行复杂核函数计算时优势很明显，能够大大简化模型和计算量。
 &emsp;&emsp;  而LR则每个点都需要两两计算核函数，计算量太过庞大。

#### 5. LR和SVM在实际应用的区别

 &emsp;&emsp;  根据经验来看，对于小规模数据集，SVM的效果要好于LR，但是大数据中，SVM的计算复杂度受到限制，而LR因为训练简单，可以在线训练，所以经常会被大量采用。

<br>

## 5、决策树为什么不需要归一化

 &emsp;&emsp;  因为数值缩放不影响分裂点位置，对树模型的结构不造成影响。

<br>

## 6、过拟合的原因与解决方案

### 原因

1.  训练集的数量级和模型的复杂度不匹配。训练集的数量级要小于模型的复杂度；
2.  训练集和测试集特征分布不一致；
3.  样本里的噪音数据干扰过大，大到模型过分记住了噪音特征，反而忽略了真实的输入输出间的关系；
4.  权值学习迭代次数足够多(Overtraining)，拟合了训练数据中的噪声和训练样例中没有代表性的特征。

### 解决方案

1. 调小模型复杂度，使其适合自己训练集的数量级（缩小宽度和减小深度）
2. 数据增强来扩大数据集
3. 添加正则化项（L1 L2范数）
4. dropout
5. early stopping
6. 集成学习
7. 清洗数据，将一些脏数据去除，去除无效值和缺失值

<br>

## 7、L0、L1、L2范数的区别

 &emsp;&emsp;  **L0范数是指向量中非0的元素的个数**。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0即让参数W是稀疏的。 

 &emsp;&emsp;  L1范数是指向量中各个元素绝对值之和，L1也可以使权重稀疏，是L0范数的最优凸近似

 &emsp;&emsp;  L2范数被称为**岭回归**或者**权重衰减**，它使得目标函数变为凸函数，L2范数可以使得权重比较小。一般来说权重比较小意味着模型比较简单，泛化性越强。

<br>

## 8、参数稀疏的好处

 **1. 特征选择(Feature Selection)**

 &emsp;&emsp;  大家对稀疏规则化趋之若鹜的一个关键原因在于它能实现特征的自动选择。一般来说，样本$\ x_i$ 的大部分元素（也就是特征）都是和最终的输出$\ y_i$ 没有关系或者不提供任何信息的，在最小化目标函数的时候考虑$\ x_i$ 这些额外的特征，虽然可以获得更小的训练误差，但在预测新的样本时，这些没用的信息反而会被考虑，从而干扰了对正确$\ y_i$ 的预测。稀疏规则化算子的引入就是为了完成特征自动选择的光荣使命，它会学习地去掉这些没有信息的特征，也就是把这些特征对应的权重置为0。

**2. 可解释性(Interpretability)**

 &emsp;&emsp;  另一个青睐于稀疏的理由是，模型更容易解释。例如患某种病的概率是y，然后我们收集到的数据x是1000维的，也就是我们需要寻找这1000种因素到底是怎么影响患上这种病的概率的。假设我们这个是个回归模型：y=w1*x1+w2*x2+…+w1000*x1000+b（当然了，为了让y限定在[0,1]的范围，一般还得加个Logistic函数）。通过学习，如果最后学习到的w*就只有很少的非零元素，例如只有5个非零的wi，那么我们就有理由相信，这些对应的特征在患病分析上面提供的信息是巨大的，决策性的。也就是说，患不患这种病只和这5个因素有关，那医生就好分析多了。但如果1000个wi都非0，医生面对这1000种因素，累觉不爱。



**参考链接**

- [机器学习中样本不平衡的处理方法](https://zhuanlan.zhihu.com/p/28850865)
- [【机器学习】Linear SVM 和 LR 的联系和区别](https://blog.csdn.net/haolexiao/article/details/70191667)
- [过拟合（定义、出现的原因4种、解决方案7种）](https://blog.csdn.net/NIGHT_SILENT/article/details/80795640)