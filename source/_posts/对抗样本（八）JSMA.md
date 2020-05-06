---
title: 对抗样本（八）JSMA
date: 2020-04-03 13:08:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,网络蒸馏"
description: "对抗样本（八）JSMA"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## 1.论文题目

 &emsp;&emsp; **The Limitations of Deep Learning in Adversarial Settings**

## 2.论文时间

 &emsp;&emsp; **2015年**

## 3.论文文献

 &emsp;&emsp; https://arxiv.org/abs/1511.07528



# 二、论文背景及简介

 &emsp;&emsp; 在本篇论文中，作者基于对输入和输出的匹配的理解，提出了一个新的生成对抗样本的方法，作者利用了模型对输入样本的**Jacobian矩阵**，同时利用了输入样本的特征。作者定义了一个测量方法用来描述样本类别的脆弱性。作者定义了一个对输入和目标类别的距离计算的方法。

<br>

# 三、论文内容总结

- 作者提出了一个**从样本出发，找到适合的扰动，来进行攻击**的攻击方法。
- 攻击方法分为三步：**计算Jacobian矩阵，根据Jacobian矩阵得到saliency map，根据saliency map定位要变化的输入特征**。
- saliency map表示的是输入样本的特征对类别标签的贡献度
- 定义了一种测量 一个样本由原始标签攻击成目标标签的难易程度的标准Hardness measure
- 借助saliency map，定义了一种对输入样本被攻击成某一标签的难易程度的预测的标准

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

<br>

# 四、论文主要内容

## 1. Introduction

 &emsp;&emsp; 之前的对抗攻击的工作都是使用梯度来进行的。在这片论文中，提出了一个新奇的方法，作者计算了**从输入到输出的一个直接映射**，得到了一个**明确的对抗目标**。而且，这个方法**只需要修改一小部分的输入特征**，就能够得到对抗样本。而且，这个方法也可以使用**启发式的搜索方法**。

 &emsp;&emsp; 值得注意的是，我们这个方法是，**构建了一个输入扰动到输出变化的矩阵，也就是首先得到了这个矩阵，再添加对应的扰动使得攻击成功**。而之前提到的方法是根据输出的变化而得到输入的扰动，这两个过程是相反的。我们对输入变化如何影响DNN输出的理解源于对前向导数的评估。

 &emsp;&emsp; 我们引入了Jacobian矩阵，前向导数被用来构建对抗性saliency map，用来表示输入特征。saliency map是基于前向导数的多用途工具，会考虑到对抗目标，这在对抗扰动的选择上给予了更多的选择。

 &emsp;&emsp; 在我们的工作中，我们考虑到了以下的问题：

- 对抗攻击所需要的最小的knowledge是什么？
- 怎么样才能识别对抗样本？
- 人类是怎么识别对抗样本的？

 &emsp;&emsp; 该方法在LeNet的手写数字是手写识别任务上达到了97.1%的攻击成功率，而且只修改了4.02%的输入特征。且每个样本生成所花费的时间不到1s。

## 2. Taxonomy Of Threat Models In DL

 &emsp;&emsp; 这一节主要介绍了DL的基本知识以及对抗目标还有对抗能力（白黑盒攻击）。

 &emsp;&emsp; 对抗目标可以分为四类：1.减少输出的confidence，2.无目标攻击，3.有目标攻击，4.**Source/target 攻击**，即使得特定的输入被分类成特定的输出。住：本文的方法就属于source/target 攻击。

## 3. Approach

 &emsp;&emsp; 在这一节主要介绍了作者的对抗攻击方法。在攻击方法中需要**通过DNN的前向导数构建对抗性saliency map识别与对抗目标相关的输入特征集**。这种方法**既可以用于监督学习也可以用于非监督学习**。

### A. Studying a Simple Neural Network

 &emsp;&emsp; **简单的模型有助于理解算法的思想**，所以先以简单的模型为例进行讲解。

 &emsp;&emsp; 作者训练了一个神经网络去拟合函数$\ F(X)=x_1 \wedge x_2$ ，网络输入为$\ (x_1,x_2)，x \in [0,1]$ ，输出为$\ \{0,1\}$ ，交运算中采用四舍五入的运算规则，即$\ 0.8 \wedge 0.6 = 1$ 。我们现在要对这样一个模型来进行攻击，其问题可以表示为：
$$
arg\ min_{\delta_X}||\delta_X|| \quad s.t.\ F(X+\delta_X) = Y^*
$$
![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/1.png?raw=true)

 &emsp;&emsp; 我们用函数F的Jacobian矩阵定义前向导数，对该问题而言，其矩阵为：
$$
\nabla F(x) = [\frac{\partial F(x)}{\partial x_1},\frac{\partial F(x)}{\partial x_2}]
$$
 &emsp;&emsp; 这个向量的每一个元素都是可计算的。作者得到的前向导数的值的图像如下：

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/2.png?raw=true)

 &emsp;&emsp; **前向导数可以告诉我们哪些输入区域不太可能产生对抗样本**。值越小，越不容易产生。也就是说，当我们去生成对抗样本时，我们要关注那些能够得到更大的前向导数的那部分特征。

### B. Generalizing to Feedforward Deep Nerual Networks

 &emsp;&emsp; 在这一节，我们要将上面的思想拓展到复杂的网络中去。

 &emsp;&emsp; 作者所使用的网络以及符号如下：

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/3.png?raw=true)

 &emsp;&emsp; 其算法过程如下：

1. 定义网络输入$\ X$ ，网络输出$\ Y$ ，对抗样本$\ X*$ ，攻击目标$\ Y*$ ，最大扰动$\ \gamma$ ，单次特征扰动参数$\ \theta$ 。
2. 首先**计算前向导数**：

$$
\nabla F(X) = \frac{\partial F(x)}{\partial x} = [\frac{\partial F_j(x)}{\partial x_i}]_{i \in [1...M],j\in[1...N]}
$$

 &emsp;&emsp; 我们可以通过链式法则，很容易的就能够得到上面的Jacobian矩阵

3. **生成saliency maps**，通过saliency maps找到能够最大效率扰动样本的输入特征点。saliency maps是一个根据目标函数自定义的一个矩阵，对于我们的对抗目标而言，对于输入$\ X$ ，攻击目标$\ t$ ，我们的目标是让$\ F_t(x)$ 增大，而使$\ F_j(x) \ j \ne t$ 减小，知道网络将样本分类为$\ t$ 。我们可以通过增加样本特征值，来实现改变其输出概率。我们可以定义该问题的saliency map $\ S(X,t)$ 为：

$$
S(X,t)[i] = 
\begin{cases}
0 & & if\ \frac{\partial F_t(x)}{\partial X_i}<0\ or\ \sum_{j\ne t}\frac{\partial F_j(x)}{\partial X_i}>0\\
\frac{\partial F_t(x)}{\partial X_i}·|\sum_{j\ne t}\frac{\partial F_j(x)}{\partial X_i}| & & otherwise \\
\end{cases} \\
$$

 &emsp;&emsp; 第一行条件的意思为：只有当$\ \frac{\part F_t(x)}{\part X_i}>0$ 时，该单元对增大$\ F_t(x)$ 才有帮助，当$\ \sum_{j\ne t}\frac{\part F_j(x)}{\part X_i}<0$ 时，该单元才能用来减小其他单元的概率值。

 &emsp;&emsp; 也就是说，当$\ S(X,t)[i]$ 越大时，增大该特征的值，对抗攻击成功率越大。

 &emsp;&emsp; saliency map可以定义多种形式，这对算法是有影响的，作者还提出了下面的一种saliency map：
$$
S(X,t)[i] = 
\begin{cases}
0 & & if\ \frac{\partial F_t(x)}{\partial X_i}》0\ or\ \sum_{j\ne t}\frac{\partial F_j(x)}{\partial X_i}《0\\
|\frac{\partial F_t(x)}{\partial X_i}|·\sum_{j\ne t}\frac{\partial F_j(x)}{\partial X_i} & & otherwise \\
\end{cases} \\
$$
 &emsp;&emsp; 这一个就跟上面那一个刚好相反，也就是说，当$\ S(X,t)[i]$ 越大时，减小该特征的值，对抗攻击成功率越大。

4. **调整样本**，在确定了要扰动的特征后，我们要对其特征值进行扰动，通过参数$\ \theta$ 来修改特征值（相加，例如可取$\ \theta=1\ or\ -1$ 。
5. 不断地迭代这个过程，知道修改的总特征值大于最大限制$\ \gamma$ ，或者迭代次数达到了最大迭代次数。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/4.png?raw=true)

## 4. Application Of The Approach

 &emsp;&emsp; 在这一节，我们以手写数字识别任务为例，看一下该算法如何应用。

 &emsp;&emsp; 在手写识别任务中，每次选取了最大的两个像素点，更改完后，从特征集$\ \Gamma$ 中删除，取最大迭代次数为$\ |\frac{784·\gamma}{2·100}|$ ，$\ \theta$ 可根据我们的生成策略设置$\ 1\ or \ -1$ 。

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/5.png?raw=true)

 &emsp;&emsp; 同时，使用该方法，我们可以将一个空白的输入图像一步步的修改为一个分类器可以辨别的图像，以下是进行这样操作得到的类别从0--9的图像。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/6.png?raw=true)

## 5. Evaluation

 &emsp;&emsp; 在这一节，主要是在实验的基础上回答了三个问题：

- 上面的算法能够把每一个样本都变成对抗样本？
- 我们怎样区分样本的脆弱性？
- 为什么人类能区分对抗样本？（说明了改方法生成的对抗样本在低扰动时可以不被人察觉）

### A.  对抗样本生成率

 &emsp;&emsp; 作者在手写数字识别任务上进行了实验，分成3组数据，分别来自于训练集、验证集、测试集，每组10000张图片，为每一张图片生成9个对抗样本（9个其他类别）。这样每一组就会得到90000张对抗样本。设置$\ \gamma = 14.5\%$ ，$\ \theta = 1$ （采用像素增加的方式）。实验得到，97.1%的对抗样本可以以小于14.5%修改率的结果得到。值得注意的是，输入图像时归一化到[0,1]内的，所以**每次处理像素，都是将像素设置为最高值**。

 &emsp;&emsp; 当作者采用$\ \theta=-1$ （采用像素减少的方式）时，其攻击成功率只有64.7% ，这可能是因为降低像素会减少输入图片的信息，这让网络难以提取信息，也难以进行分类。

### B. 样本的脆弱性

 &emsp;&emsp; 之前的实验，我们可以得到大约2.9%的图像没有被攻击成功，这表明有一些图像很难被攻击。

 &emsp;&emsp; 我们需要研究哪些source-target类别对是最容易被攻击的或者最不容易被攻击的，同时提出了一个hardness测量标准来量化这些现象。通过这些现象我们可以得到一个防御的方法。

#### 1）类别对研究

 &emsp;&emsp; 作者根据攻击情况，得到了下面的图像，行表示初始类别，列表示目标类别，越黑表示攻击成功率越高。

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/7.png?raw=true)

 &emsp;&emsp; 从图像中可以看出，初始类别0、2、8不容易被攻击，初始类别1、7、9容易被攻击，很难制作1、7类别的对抗样本，很容易制成0、8、9类别的对抗样本。

 &emsp;&emsp; 作者根据每个类别对攻击成功时的平均扰动值，得到了下面的图像

![8](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/8.png?raw=true)

 &emsp;&emsp; 可以看到，有着更低扰动的类别对，其攻击成功率越高。

#### 2）Hardness标准

 &emsp;&emsp; 上面的现象驱使着我们要找到一个测量标准来量化两个类别作为source-target攻击的难易度。这对于对抗防御来说是很有用的。

 &emsp;&emsp; 该标准是**通过归一化一个类别对相对于其成功率的平均扰动值**得到的。
$$
H(s,t) = \lmoustache_{\tau}\epsilon(s,t,\tau)d \tau
$$
 &emsp;&emsp; 其中$\ \epsilon(s,t,\tau)$ 表示在攻击成功率为$\ \tau$ 时的平均扰动。**越大表明其越难被攻击**

 &emsp;&emsp; 在实际情况下，我们可以使用有限个样本计算得到，首先我们定义K个最大可扰动值$\ \gamma_k$ ，根据每一个$\ r_k$ 生成一组对抗样本，得到改组对抗样本的攻击成功率$\ \tau_k$ ，以及平均扰动值$\ \epsilon_k$ ，我们就可以计算值：
$$
H(s,t) \thickapprox \sum_{k=1}^{K-1}(\tau_{k+1}-\tau_k)\frac{\epsilon(s,t,\tau_{k+1}+\epsilon(s,t,\tau_{k})}{2}
$$
 &emsp;&emsp; 在本次实验中，我们取$\ K=9,\gamma=[0.3,1.3,2.6,5.1,7.7,10.2,12.8,25.5,38.3]\%$ 。得到的Hardness的矩阵图为：

![9](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/9.png?raw=true)

#### 3）adversarial distance

 &emsp;&emsp; Hardness有一个很致命的问题，那就是他是在对抗攻击之后计算的，而并不是在对抗攻击前就进行预测。也即是说我们需要一个测量方法，我们预测其输入样本的脆弱性。

 &emsp;&emsp; 基于直觉和saliency map，作者得到了测量方法，记为$\ A(X,t)$ 表示样本$\ X$ 到类别$\ t$ 的对抗距离：
$$
A(X,t) = 1-\frac{1}{M}\sum_{i \in 0...M}1_{S(X,t)[i]>0}
$$
 &emsp;&emsp; $\ 1_E$ 表示，只有当$\ E$ 为true，该式子才为1，否则为0。$\ A(X,t)$ 越大，说明$\ X$ 越难被攻击为类别$\ t$ 。

![10](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AB%EF%BC%89JSMA/10.png?raw=true)

 &emsp;&emsp; 同时作者用该攻击来表示网络$\ F$ 的鲁棒性：
$$
R(F) = min_{(X,t)} A(X,t)
$$
 &emsp;&emsp; 该值越大，鲁棒性越好。



## 5. Discussion

 &emsp;&emsp; 这篇论文假设网络是可导的，同时该论文以LeNet作为Baseline，认为攻击成功LeNet一定可以攻击那些深度模型（博主觉得有点不妥）。

 &emsp;&emsp; 与先前的工作相比，这篇论文**从样本出发，找到适合的扰动，来进行攻击**。同时**其攻击方式扰动的输入特征数目较少**。

 &emsp;&emsp; **生成Jacobian矩阵的代价很大，该方法的速度也比较慢**