---
title: 文本对抗攻击之Word-Level笔记系列（二）
date: 2021-03-01 05:21:00
tags:
 - [文本对抗攻击]
 - [Word-Level]
 - [论文笔记]
categories: 
 - [深度学习,文本对抗]
keyword: "深度学习,文本对抗,论文笔记"
description: "文本对抗攻击之Word-Level笔记系列（二）"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、Breaking NLI Systems with Sentences that Require Simple Lexical Inferences

## 1. Paper Information

> 时间：2018年
>
> 关键词：Adversarial Attack，NLI
>
> 论文位置：https://www.aclweb.org/anthology/P18-2103.pdf
>
> 引用：Glockner M, Shwartz V, Goldberg Y. Breaking NLI systems with sentences that require simple lexical inferences[J]. arXiv preprint arXiv:1805.02266, 2018.

## 2. Motivation

 &emsp;&emsp; NLI系统的泛化性能不强，无法掌握许多需要词汇和世界知识的简单推论。本文通过创建了一个简单的NLI数据测试集来证明。

## 3. Main Arguments

 &emsp;&emsp; 我们创建了一个新的NLI测试集，显示了在**需要词汇和世界知识的推理**中，SOTA模型的缺陷。新的样本比SNLI测试集更简单，包含的句子与训练集中的句子最多相差一个单词。然而，在用SNLI训练的系统中，新测试集的性能明显较差，这表明这些系统的泛化能力有限，无法捕捉到许多简单的推论。

## 4. Framework

### 4.1 Data Collection -- Generating Adversarial Examples

 &emsp;&emsp; 为了去捕捉NLI模型在词汇知识方面的能力，我们从SNLI训练集中提取了前提。对于每个前提，我们用不同的词替换前提中的单个词，从而生成几个假设。我们也允许一些多词名词短语(“electric guitar”)，并在需要时使用限定词和介词。

 &emsp;&emsp; 我们只关注生成蕴涵和矛盾的样本，而中性的例子可能作为副产品产生。蕴涵样本是通过将一个词替换为它的同义词或连词来生成的，矛盾样本是通过将一个词替换为互斥的下义词和反义词来生成的(见表1)。

 &emsp;&emsp; 作者通过网上信息找了一些替换词，来进行替换，于是就形成了新的数据集。

## 5.Result

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/1.png?raw=true)

## 6. Argument



## 7. Further research







# 二、Generating Natural Language Adversarial Examples

## 1. Paper Information

> 时间：2018年
>
> 关键词：Adversarial Attack，Text Classfication，Word-based， Score-based，GA
>
> 论文位置：https://arxiv.org/pdf/1804.07998.pdf?source=post_page---------------------------
>
> 引用：Alzantot M, Sharma Y, Elgohary A, et al. Generating natural language adversarial examples[J]. arXiv preprint arXiv:1804.07998, 2018.

## 2. Motivation

 &emsp;&emsp;  在图像领域中，我们可以得到令人无法察觉的扰动，但是文本领域不行，无论是word-level、char-level还是sentence-level，扰动变化都是很明显的，一个单词的替换都会较大的改变句子的语义。在这些挑战下，在黑盒模型进行攻击还是很难的。

## 3. Main Arguments

 &emsp;&emsp;  我们使用了一种population-based优化方法来生成语义上和语法上都近似的对抗样本。我们采用了遗传算法，基于Threat Model的score来进行攻击。

## 4. Framework

 &emsp;&emsp;  我们算法的目标是在保证语义相似性的前提下，最小化修改单词的数量，为了实现这个目标，我们采用了遗传算法来进行。

 &emsp;&emsp;  遗传算法是一种population-based算法，每一次迭代被称之为一代（generation），每一代成员的质量是通过fitness函数来进行评估，更优的成员被选择来生成下一代。我们主要是通过变异（mutation）和杂交（crossover）来实现下一代的生成。杂交是通过多个父代通过特殊的规则来生成子代，变异是通过一个父代来生成子代，是为了增加种群的多样性。

### 4.1 子程序Perturb

 &emsp;&emsp;  在介绍我们整个优化程序之前，先介绍一下我们的子程序Perturb。**这个子程序接受一个输入句子$\ x_{cur}$** ，这个句子可以使修改后的句子，也可以是原始文本。它从该句子中随机的选择一个单词$\ w$ ，然后选择一个合适的替代词进行替换，该替代词需要是$\ w$ 的同义词，符合上下文语义，并且可以使修改后的文本在目标label的预测score增加。主要包括以下几个步骤：

1. 首先，我们得到被选中的单词$\ w$ 的N个近邻，在这里我们使用了GloVe词向量，通过欧氏距离进行选择，被选择的词与$\ w$ 的欧氏距离需要小于$\ \delta$ ，同时要确保他是同义词，这里使用了counter-fitting方法 。
2. 之后，我们使用Google的10亿的语言模型来剔除那些不符合上下文语境的单词，同时基于该语言模型的预测score来对这些候选替换词进行排名，然后，只留下分数最高的K个候选词。
3. 从K个候选词中，我们选择一个能够使目标label的预测概率最大的那一个作为替换词。
4. 最后，被选择的词替换掉单词$\ w$ ，**子程序Perturb返回修改后的句子**。

 &emsp;&emsp;  该子程序在输入句子上选择需要被替换的单词时，我们是根据每个单词的同义词数目来进行随机挑选的，数目越大，被挑选的概率越大，这是为了让我们的算法有足够大的搜索空间来做出合适的修改。

### 4.2 优化程序

 &emsp;&emsp;  首先，我们调用子程序Perturb $\ \mathcal{S}$ 次，来生成第一代$\ \mathcal{P}^0$ 。针对每一代的成员的fitness函数，我们采用了目标label的预测概率，因此在评估时，我们需要访问victim model。如果，某一代中的成员的预测标签是目标标签，即攻击成功，则算法停止。否则，以fitness值作为挑选概率，从当前这一代中随机的选出两个作为parents，这两个句子对应位置的两个单词随机选择一个作为子句该位置的单词，这样就得到了一个新的子句，之后对该子句进行一次Perturb子程序，得到修改后的句子，该句子加入下一代。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/2.png?raw=true" alt="2" style="zoom: 67%;" />

## 5.Result

 ### 5.1 Setup

 &emsp;&emsp;  这里主要使用了情感分类和文本蕴含的分类任务，采用了300维的GloVe词向量。情感分类使用了IMDB（90%准确率），文本蕴含使用了SNLI（83%准确率，文本蕴含任务的攻击是修改假设）。

 &emsp;&emsp;  超参选择，迭代次数$\ G=20$ ，$\ S=60,N=8,K=4,\delta=0.5$ ，在两个任务中，超过20%、25%的扰动比例算攻击失败。

### 5.2 攻击成功率

 &emsp;&emsp;  分别采样了1000、500个被正确分类的样本进行攻击实验。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/3.png?raw=true)

## 6. Argument



## 7. Further research









# 三、Universal Adversarial Attacks on Text Classifiers

## 1. Paper Information

> 时间：2019年
>
> 关键词：Adversarial Attack，Gradient
>
> 论文位置：https://infoscience.epfl.ch/record/264189/files/pdf.pdf
>
> 引用：Behjati M, Moosavi-Dezfooli S M, Baghshah M S, et al. Universal adversarial attacks on text classifiers[C]//ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019: 7345-7349.

## 2. Motivation

 &emsp;&emsp; 我们发现，在每个输入序列的开头插入哪怕是一个对抗性的单词，文本分类器的准确率将会大打折扣。基于梯度，我们试图生成一种文本的通用扰动。

## 3. Main Arguments

 &emsp;&emsp; 我们想要找到一个序列$\ w$ ，通过将其加入到分布为$\ P(x)$ 的原始样本中，来愚弄分类器。其问题可以描述为：

 &emsp;&emsp; 对于输入$\ x$ ，输出$\ l$ ，对抗序列$\ w = w_1w_2...w_m$ 。我们试图构建对抗样本$\ x'$ ：
$$
x' = w \oplus_{k} x = x_1...x_k;w_1...w_m;w_{k+1}...x_n \tag{1}
$$
 &emsp;&emsp; 其中，$\ \oplus$ 表示插入操作，$\ k$ 表示插入的位置。

 &emsp;&emsp; 因此，我们要解决的问题是：
$$
\hat{w} = \arg \max_w \mathbb{E}_{x \sim P(X)}[loss(l,f(x'))] \tag{2}
$$
 &emsp;&emsp; 对于目标攻击，我们要解决的则是：
$$
\hat{w} = \arg \min_w \mathbb{E}_{x \sim P(X)}[loss(l',f(x'))] \tag{3}
$$

## 4. Framework

 &emsp;&emsp; 我们拟使用梯度下降/上升方法来找到$\ \hat{w}$ ，这取决于我们是目标攻击还是非目标攻击。在每次梯度下降/上升的迭代中，我们都会对$\ w$ 中的单词的词向量进行更新，得到的向量再映射到单词空间中，即我们要找到$\ w_i' $ ：
$$
w_i' = \arg \min_{w_i' \in V} \ \cos(\text{emb}(w_i'), (\text{emb}(w_i) + \alpha r_i))
$$
 &emsp;&emsp; 其中，$\ r_i = \nabla_{\text{emb}(w_i)} \text{loss}(l,f(x'))$ 是在单词$\  w_i$ 上的梯度。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/4.png?raw=true" alt="4" style="zoom:50%;" />

 &emsp;&emsp; 在无目标攻击中，我们需要朝着梯度的方向移动来最大化loss，所以此时$\ \alpha$ 是整数。在目标攻击中，我们需要朝着梯度反方向移动来最小化loss，所以此时$\ \alpha$ 是负数。算法伪代码如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/5.png?raw=true" alt="5" style="zoom:50%;" />

## 5.Result

### 5.1 Performance

 &emsp;&emsp; 在文本的句首添加对抗性单词，在无目标攻击下的攻击效果

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/6.png?raw=true)

 &emsp;&emsp; 在不同的模型架构下的攻击效果

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/7.png?raw=true" alt="7" style="zoom:50%;" />

 &emsp;&emsp; 添加的位置不同的攻击效果

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/8.png?raw=true" alt="8" style="zoom:50%;" />

 &emsp;&emsp; 之后，我们尝试目标攻击，在句首添加对抗性单词，攻击效果如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%BA%8C%EF%BC%89/9.png?raw=true" alt="9" style="zoom:50%;" />

## 6. Argument

 &emsp;&emsp; 通过将未知单词插入到文本中，利用梯度迭代来寻找合适的单词，方法很好。

## 7. Further research







