---
title: 对抗样本（十四）Hot&Cold
date: 2020-04-03 13:14:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,Hot&Cold"
description: "对抗样本（十四）Hot&Cold"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89Hot&Cold/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **Adversarial Diversity and Hard Positive Generation** 

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2016年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp; https://arxiv.org/pdf/1605.01775.pdf

<br>

# 二、论文背景及简介

 &emsp;&emsp; 这篇文章提出了一个新的量化对抗样本的测量方法，**psychometric perceptual adversar- ial similarity score (PASS)** ，介绍了h**ard positive genneration**（hard表示困难，即分类概率较小，loss较大的图像；positive表示正样本）的概念，并使用了一组具有多样性的对抗扰动来进行数据增强。介绍了一个新的**hot/cold方法**来生成对抗样本，该方法为每个图片提供多个对抗扰动。**我们的新方法产生的扰动通常对应于语义上有意义的图像结构，并且允许更大的灵活性来缩放扰动幅度，从而增加对抗性图像的多样性**。

<br>

# 三、论文内容总结

- 提出了一种新的量化对抗样本的测量方法，叫做**Perceptual Adversarial Similarity Score (PASS)**，该方法比$\ L_p$ 范数**更接近于人类感知**
- 提出了一个新的生成对抗样本的方法，**hot/cold方法**，能够生成大量的具有多样性的对抗样本
- 使用对抗样本来增强数据集可以提高模型的准确率和鲁棒性，作者证明，使用非最小的扰动而形成的hard postitive更好一些，更有效率。

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

# 四、论文主要内容

## 1. Introduction

 &emsp;&emsp; 对抗样本的各种特性说明，我们仅仅用训练样本和算法的组合来训练网络是不够的。

 &emsp;&emsp; 这篇论文的主要贡献如下：

- 提出了一种新的量化对抗样本的测量方法，叫做Perceptual Adversarial Similarity Score (PASS)，该方法比$\ L_p$ 范数更接近于人类感知
- 提出了一个新的生成对抗样本的方法，hot/cold方法，能够生成大量的具有多样性的对抗样本
- 使用对抗样本来增强数据集可以提高模型的准确率和鲁棒性，作者证明，使用非最小的扰动而形成的hard postitive更好一些，更有效率。

## 2. Related Work

 &emsp;&emsp; 简要的介绍一下先前的工作

## 3. PASS

 &emsp;&emsp; 有许多不同的量化对抗样本的方法，比如比较常用的$\ L_p$ 举例。在Feature adversary中，作者证明到$\ L_p$ 举例并不能很好的符合人类的感知。对抗样本应该根据明显的差异来进行量化。然后，**对不可感知的自然解释应该允许许多变换，包括小的平移和旋转。而这也会导致，当某图像受到明显的扰动时，对网络来说，这仍然应该是一个可信的样本，是一个网络不应该分类错误的样本。**拿对生物特征人脸识别系统的攻击举例，对人脸的观察角度不同，会得到一个人脸的不同图像，如果扭曲比较大，比如：得到的脸非常扭曲或者噪声很大，但是观察者除了发现该图片有问题外，仍然可以识别他的身份，即当扰动较大时，网络也应该识别正确。

 &emsp;&emsp; 那么，为了量化，  一张图片扰动到什么程度才是一个对抗样本，我们找到了一种**心理测量距离**，这种距离**不仅仅考虑了元素相似性，而且考虑到了图片中的可信图片问题（上面说的问题）**。

 &emsp;&emsp; 为了得到这样的距离，我们会进行两个步骤。首先将扰动的图像与原图像对齐，然后测量对齐图像的相似性。

 &emsp;&emsp; 第一步，由于图像之间潜在的辐射和噪声差异，简单的相关或基于特征的对齐可能不会很好地工作。我们**通过最大化对抗样本与原始图像之间的enhanced correlation coefficient (ECC)来进行操作**。记$\ \psi(\tilde{x},x)$ 表示从对抗样本到原始图像的单对应性转变，其对应的单对应性矩阵为$\ H$ ，那么我们需要优化以下的目标函数：
$$
argmin_H ||\frac{\overline{x} }{||\overline{x}||} - \frac{\overline{\psi(\tilde{x},x)} }{||\overline{\psi(\tilde{x},x)}||}||
$$
 &emsp;&emsp; 其中上划线表示的是图像转换为均值为0后的版本。最小化上式也就会最大化$\ ECC(\psi(\tilde{x},x),x)$ 。

 &emsp;&emsp; 第二步，就是相似性的度量。之前的工作使用$\ L_2,L_{\infty}$ 来量化相似度，但是这两种距离公式对小的几何变换太过于敏感，哪怕图像没有经过几何变换，其也不能够衡量心理测量距离的相似度。而在对其之后，在边界上的像素差异足以使得$\ L_{\infty}$ 具有一个较大的值，哪怕中间部分非常的相似。

 &emsp;&emsp; 有研究表明，人类视觉系统对结构模式的变化最为敏感，所以使用了structural similarity（SSIM）index来作为有损图像压缩的目标，**SSIM就是基于结构和亮度差异来量化图像的相似性。**

 &emsp;&emsp; 给定两个图像$\ x,y$ ，让$\ L(x,y),C(x,y),S(x,y)$ 分别表示两个图象的亮度、对比度和结构距离，定义为：
$$
\begin{split}
L(x,y)&=[\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}]\\
C(x,y)&=[\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}]\\
S(x,y)&=[\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}]\\
\end{split}
$$
 &emsp;&emsp; 其中，$\ \mu_x,\sigma_x,\sigma_{xy}$ 分别表示加权平均数、方差和协方差，$\ C_i$ 是一个防止奇异性的常数。那么，SSIM index（RSSIM）可以定义为：
$$
RSSIM(x,y) = L(x,y)^{\alpha}C(x,y)^{\beta}S(x,y)^{\gamma}
$$
 &emsp;&emsp; 其中，$\ \alpha,\beta,\gamma$ 用来衡量三者的相对重要性。在论文中，设定$\ \alpha=\beta=\gamma=1$ ，设置$\ \sigma=1.5$ 的11x11的权重核。SSIM通过对整个像素取RSSIM的平均来得到，即：
$$
SSIM(x,y) = \frac{1}{m}\sum_{n=1}^mRSSIM(x_n,y_n)
$$
 &emsp;&emsp; 我们将光度不变的单应变换对齐与SSIM相结合，得到PASS
$$
PASS(\tilde{x},x) = SSIM(\psi(\tilde{x},x),x)
$$
 &emsp;&emsp; PASS作为一个相似性度量来量化错误分类图像的对抗性。Szegedy、Goodfellow等人都提到了一个隐式的假设，但并没有给出明确的定义，即：对抗扰动必须是不可察觉的。而依据$\ L_p$ 而得到的数学定义并不们能够很好的反应这一假设。而PASS可以，我们可以用它来明确这个约束。记$\ y$ 为图像$\ x$ 的index，记$\ f$ 为分类器，记$\ \tau \in [0,1]$ 表示PASS阈值。那么对抗样本可以定义为：
$$
argmin_{d(x,\tilde{x})} \tilde{x}:f(\tilde{x}) \ne y \quad s.t. \ PASS(x,\tilde{x}) \ge \tau
$$
 &emsp;&emsp; 其中$\ d(x,\tilde{x})$ 表示非相似性的距离，比如$\ 1-PASS(x,\tilde{x})$ 或者$\ ||x-\tilde{x}||_p$ 减去 沿对抗样本生成算法学习方向的潜在约束。注意：$\ \tau$ 在不同的网络和数据集上可能是不同的，但是对于一个固定的领域来说，这给定了一种量化的对抗样本阈值。Hard positive也受到PASS阈值的约束，但是其不需要是最相似的那一个。

## 4. Adversarial Example Generation

 &emsp;&emsp; 在这一节，首先讨论了一个现存的生成对抗样本的方法，之后介绍了一个新颖的生成对抗样本的方法以及hard postive generation和实现的细节

 &emsp;&emsp; 记$\ \theta$ 表示模型的参数，$\ x \in [0,255]^m$ ，$\ y$ 为$\ x$ 的标签，$\ J(\theta,x,y)$ 表示网络的损失函数，$\ f$ 表示分类器，$\ B_l(·)$ 表示反向传播操作。

 &emsp;&emsp; 对于给定的$\ x$ ，我们的目标是，添加一个扰动$\ \eta$ ，得到对抗样本$\ \tilde{x}=x+\eta$ 。为了生成hard positive，我们会简单的设置$\ \eta$ 为一个$\ \ge 1$ 的常数。

### 4.1 Fast Gradient Value

 &emsp;&emsp; 我们把我们的研究放到了FGS方法中。FGSM的一个很明显的拓展就是直接使用梯度来得到扰动，而不是梯度的方向，作者把他称之为FGV。作者发现，FGV会生成一些不同的对抗扰动。其对应的扰动方向为：
$$
\eta_{grad} = \bigtriangledown_xJ(\theta,x,y)
$$
 &emsp;&emsp; 而上面的式子并没有忽略梯度的大小，FGSM采用sign忽略了梯度的大小。FGV更好一些。但是，FGS和FGV都只能生成一个对抗样本，那么我们可以生成更多的对抗样本吗？

### 4.2 Hot/Cold Approach

 &emsp;&emsp; FGS和FGV是通过遵循损失梯度的方向来减少对感兴趣的类别的响应。为了生成hard positive（能够增加类别之间的过拟合决策边界），那么，我们很自然的就会考虑其他层的导数，生成能够向特定目标类别移动从而改变分类的对抗样本。

 &emsp;&emsp; 我们的idea与这篇论文相关A. Mahendran and A. Vedaldi. Understanding deep image representations by inverting them.在这片论文中，引入了叫做image inverting的方法，其通过一个在倒数第二层（生成logits那一层）的one-hot的特征向量，来重建一个能够使给定类别的loss最小化图像。

 &emsp;&emsp; 我们假设，在倒数第二层反转一个one-hot向量，最终会在较低层创造一个专有于所选hot类的特征，以及负责中和倒数第二层中由零表示的其他类的其他特征层（非hot类）。我们通过添加“cold”类来扩展这个概念，以进一步减少当前类的职责。特别的，我们会为倒数第二层制作特征。在这一层，每一个值仍然关联到一个特定的输出类，所以我们会在输入图片空间定义一个方向，来帮助我们朝着目标类（hot类）移动，朝着原始类别（cold类）相反的方向移动。

 &emsp;&emsp; 记$\ h(x)\in R^n$ 表示神经网络导数第二层对应输入图像$\ x$ 的特征，其label为$\ y$ 。**接下来，我们基于$\ h(x)$ 来构建hot/cold的特征向量$\ w_{hc}$ 。首先，我们定义一个目标类$\ \tilde{y}\ne y$ 作为hot类，我们会给输入图像增加一些专有于$\ \tilde{y}$ 类的特征$\ |h_{\tilde{y} }(x)|$ 。第二步，我们定义类别$\ y$ 为cold类，通过值$\ -h_y(x)$ 来使得图片远离原始类别。**因此，构建得到的特征向量为：
$$
w_{hc}(x) = 
\begin{cases}
|h_j(x)| & & j=\tilde{y}\\
-h_j(x) & & j=y \\
0 & & otherwise
\end{cases} \\
$$
 &emsp;&emsp; 为了能够同时针对输入图像$\ x$ 的相似类和不同类，我们使用标量值$\ h_j(x)$ 作为热类。最后，我们会使用反向传播来估计需要的图像变化，通过计算$\ \eta_{hc} = B_l(w_{hc})$ 并根据我们构建的特征向量$\ w_{hc}$ 进行移动，其中操作符$\ B(·)$ 是对反向传播到图像层的导数的近似值。第4.3节讲解了如何在中间层进行。虽然任何一个针对hot类的正值、针对cold类的负值在提供对抗样本方向上都是有益的，但是，我们发现，实际上，从原始图像上提取到的特征向量得到的值表现得更好，他们自然的捕获了相应类别特征的相对比例

 &emsp;&emsp; 使用这种新方法，我们可以明确地朝着目标类的方向移动，为每个输入图像获得几个不同的对抗方向。这大大增加了可用于训练的对抗性多样性。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89Hot&Cold/1.png?raw=true)

### 4.3 Implementation Details

 &emsp;&emsp; 要在深层执行特征表示的反向传播，有两种选择。第一种，我们可以创建截断的网络，使目标层位于每个网络的顶部，然后使用反向方法执行感兴趣的反向传播。这种方法很麻烦。第二种方法是修改backward方法以直接从任何指定的特征表示进行backpropagate。通过简单地删除负责检查指定起点是否为顶层的几行代码，我们可以使用单个网络进行实验。这就是我们如何实现反向传播算子$\ B_l（·）$。

 &emsp;&emsp; 我们提出的一种新的获取对抗图像的方法可以计算任意层的特征导数，并允许我们获得相对于给定输入图像定义不同方向的扰动。我们可以沿着这些方向搜索导致错误标记的最近扰动，以获得对抗样本，或者进一步扩展此搜索以获得另外的hard positive。由于我们将给定的方向作为扰动应用于输入图像，因此通过将扰动按越来越大的值缩放并将其添加到原始图像中，原始图像将沿着该方向移动越来越远。为了有效地发现该方向最近的对抗点，我们采用了一种增加步长的线搜索技术。为了寻找对抗样本，我们在行搜索的最后一部分应用二进制搜索来搜索可能的最小对抗扰动。最后，我们要强调的是，我们生成的所有图像在[0，255]中都有离散的像素值。

## 5. Experiments

 &emsp;&emsp; 我们实验的主要关注点就是PASS的阈值。首先，我们会在LeNet/MINST上生成对抗样本，基于$\ L_2,L_{\infty}$ 和PASS距离来进行对比。之后，我们会证明，当PASS被应用到复杂的图像上时，会发现，在不同的领域，好的阈值时不同的。最后，我们证明了，当我们使用带有hard positive的图像的训练集重新训练LeNet时，其表现能力得到了增强。

### 5.1 MNIST - Adversarial Metrics

 &emsp;&emsp; 以下，是由不同类型的方法生成得到的对抗样本在各种评分标准上的分布

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89Hot&Cold/2.png?raw=true)

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89Hot&Cold/3.png?raw=true)

### 5.2 MINST-Training with Hard Positives

 &emsp;&emsp; 我们发现，使用一组hard positive来fine-tuning网络，会提高它的鲁棒性和准确率。作者训练了3个不同的LeNet/MNIST，每一个使用了60K的训练样本。我们会为每一个网络生成对抗样本以及不同类型的hard poritive image。然后从这些数据中随机选择一部分，fine-tuning每个网络三次，这样我们就得到了9个不同的网络。其中fine-tuning过程，进行了20K个迭代。训练过程采用了10K的迭代。

 &emsp;&emsp; 在测试过程中，我们使用了标准的MNIST测试集，具有10K图片，还是用了一组留出来的70K个对抗样本和hard positive图片，这些是没有用在fine-tuning过程中的。着70K张图片包含了各种各样类型的对抗样本，从14种攻击方法中，每个选取5K个样本。着14种方法包括：FGS，FGV，9种hot/cold方法(HC-1到HC-9，其中后面的数字表示的是对扰动增大的系数的不同，且hot/cold方法采用的是导数的符号)，以及HC-1 scaled by 1.05，HC-1 scaled by 1.10。

 &emsp;&emsp; 在不同模型上的错误率如下

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%8D%81%E5%9B%9B%EF%BC%89Hot&Cold/4.png?raw=true)

### 5.3 ImageNet - Adversarial Training

 &emsp;&emsp; 作者用上面的方法来fine-tuning ImageNet上的BVLC-GoogLeNet。作者为每个类别生成15个图片，然后作者一共使用了20个不同的HC-types（从HC-1到HC-20）作为hot类，这样每个类别由300个图片，共生成了300K个图片，同时为了保证PASS分数高于0.99，从300K中得到了250K个hard positives来进行训练。同时，模型使用了预训练模型。fine-tuning进行了500K个迭代。每个迭代，batch为80，epoch为25。

 &emsp;&emsp; 我们把GoogLeNet的分数本来是top-1上30.552%的错误率，top-5上10.904%的错误率。在fine-tunning后，实现了top-1上2.07e-4%的错误率，在top-5上1.01e-4%的错误率。

## 6 Conclusion

 &emsp;&emsp; 在这篇论文中，我们介绍了一个新的扰动测量方法，这让我们能够明确的对imperceptible的程度进行量化。同时介绍了一个新的生成具备多样性的对抗样本的方法，同时展示了，利用这样的方法得到的hard positive可以被用来进行数据增强，来提高模型的鲁棒性和准确率。