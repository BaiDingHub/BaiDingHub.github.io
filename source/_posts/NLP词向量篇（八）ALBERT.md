---
title: NLP词向量篇（八）ALBERT
date: 2020-09-23 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（八）ALBERT"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

## 1. Paper Information

> 时间：2019年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://arxiv.org/pdf/1909.11942.pdf
>
> 引用：Lan Z, Chen M, Goodman S, et al. Albert: A lite bert for self-supervised learning of language representations[J]. arXiv preprint arXiv:1909.11942, 2019.

## 2. Motivation

 &emsp;&emsp; 我们都知道，增大BERT模型的尺寸会使BERT的效果变得更好，当BERT越来越大时，训练代价也会越来越大。目前，SOTA的NLP模型有着上亿的参数量，我们的GPU越来越难以承担，内存开销越来越大，训练时间越来越长，那么，**怎么才能够在保持BERT效果的情况下，减少模型的参数量呢？**本篇论文就是来解决这个问题。



## 3. Main Arguments

 &emsp;&emsp; 作者**使用了两种参数简化的方法来降低内存消耗，并增加BERT的训练速度**。同时，**使用了一个自监督Loss，来建模句子间的联系**，这种Loss使得BERT模型，在具有多个句子输入的任务上表现得更好。

 &emsp;&emsp; 这两种参数简化方法如下：

- 第一种是**factorized embedding parameterization**。通过将原来的大的embedding矩阵分解成两个小的矩阵，我们将隐藏层的大小和词向量的大小的关系解耦，这样我们就可以在增大隐藏层大小的同时，而不会过度的增加词向量矩阵的参数大小。
- 第二种是**cross-layer parameter sharing**。这个方法防止了我们的参数随着网络深度的增加而增加。

 &emsp;&emsp; 在BERT-large模型上，ALBERT的参数减少了18倍，运行时间加快了1.7倍。

 &emsp;&emsp; 为了进一步提升ALBERT的性能，我们还引入了一种用于sentence-order prediction (SOP)任务的自监督Loss。SOP任务主要用来关注句间的关系，用来解决NSP任务中的无效性的问题。

 &emsp;&emsp; 最后，本篇论文的模型在GLUE、RACE、SQuAD上都取得了SOTA的效果，但却有着更少的参数量。这个模型被称为ALBERT。

## 4. Framework

 &emsp;&emsp; 记BERT的超参数为：词向量维度为$\ E$ ，enoder的层数为$\ L$ ，隐藏层输出维度为$\ H$ ，attention head数目为$\ H/64$ ，前馈网络中的神经元的维度为$\ 4H$ ，这样看的话，attention的输入Q、K、V的维度就是$\ 4 * 64 = 256$ 。 

###  4.1 Factorized embedding parameterization

 &emsp;&emsp; 由于残差连接的关系，我们一般取$\ H = E$ ，但，这种决策无论是在建模角度还是实际角度都不是最优的。我们知道，增大$\ H$ 可以提高模型的容量，进而提升性能，如果我们能够**解开$\ H$ 和$\ E$ 的关系**，我们就可以取比较大的$\ H$ ，而对词向量矩阵的大小没影响，进而减少了大量的参数增加。

 &emsp;&emsp; ALBERT采用了词向量矩阵的因式分解，将原词向量矩阵$\ V \times H$ ，分解成两个小的矩阵$\ V \times E$ 和$\ E \times H$ 。当$\ H \gg E$ 时，这种参数简化是很重要的。

### 4.2 Cross-layer parameter sharing

 &emsp;&emsp; 跨层的参数共享有很多方法，比如只共享前馈神经网络参数，或者只共享attention的参数，或者共享所有的参数。ALBERT选择共享所有的参数。

 &emsp;&emsp; 作者对某一层的输入和输出进行L2和cos距离测量，来测算输入和输出的相似性，其结果如下：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/1.png?raw=true)

 &emsp;&emsp; 我们可以看到，当网络层数增加时，输入和输出是比较相近的，这有利于增加网络参数的稳定性。而且，相比于之前的 Cross-layer parameter sharing方法（DQE），我们最后的输入和输出并不会完全一致，也就是说，我们的方法跟DQE（Deep Equilibrium Models，用在Transformer上）是完全不同的，

### 4.3 Inter-sentence coherence loss

 &emsp;&emsp; BERT的预训练中有两个任务，MLM和NSP，RoBERTa证明了NSP用处不大，还会损失BERT的精度，作者猜想是因为NSP这个任务太简单了。NSP任务包含了topic预测和句子相关性预测，但是topic预测相比于相关性预测来说太简单了，而且与MLM重叠较大。

 &emsp;&emsp; 所以，作者就提出了一个单独基于相关性的预测loss，即句序预测（sentence-order prediction (SOP)）。

## 5. Result

 &emsp;&emsp; ALBERT-large相比于BERT-large，参数量小了18倍，只有18M，BERT-large有334M。ALBERT-xlarge（$\ H = 2048$） 有60M，ALBERT-xxlarge（$\ H = $4096） 有233M。ALBERT-xxlarge只有BERT-large的70%的参数，但是在很多下游任务上都超越了BERT-large，SQuAD v1.1 (+1.9%), SQuAD v2.0 (+3.1%), MNLI (+1.4%), SST-2 (+2.2%), and RACE (+8.4%)。在速度上，ALBERT-large比BERT-large快1.5倍，但是ALBERT-xxlarge要比BERT-large慢3倍。

### 5.1 Factorized embedding parameterization

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/2.png?raw=true)

 &emsp;&emsp; 不同的词向量维度$\ E$ 以及cross-layer sharing带来的影响：

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/3.png?raw=true)

### 5.2 Cross-layer parameter sharing

 &emsp;&emsp; 不同的共享策略带来的影响

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/4.png?raw=true)

### 5.3 Sentence-order prediction (SOP)

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/5.png?raw=true)

### 5.4 相同训练时间下的效果（大致相同的参数量）

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/6.png?raw=true)

### 5.5 其他训练策略

 &emsp;&emsp; 增加的是RoBERTa和XLNet中使用的数据集

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/7.png?raw=true)

 &emsp;&emsp; 

![8](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/8.png?raw=true)

### 5.6 SOTA

![9](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/9.png?raw=true)

![10](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AB%EF%BC%89ALBERT/10.png?raw=true)



## 6. Argument

 &emsp;&emsp; ALBERT通过两种网络压缩方法极大的简化了BERT的参数量。但如果想要超过BERT的效果，我们需要堆叠更高的深度，这样就会导致训练时间更拉跨。从压缩的角度来看，ALBERT是一个巨大的提升。但我们还需要找到一个方法，在保持性能和速度的同时压缩模型。



## 7. Further research

 &emsp;&emsp; 除了作者提到的几种方法，我们还可以使用一些其他的方法来进行加速，比如sparse attention和block attention。另外，我们可以通过难样本挖掘和更有效的训练方法来提供更强的性能。另外，作者提到，在词向量中，可能会有很多维度尚未被当前的模型挖掘，而如果能进一步处理这些问题，可以进一步的提升模型的性能。