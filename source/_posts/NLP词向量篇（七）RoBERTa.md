---
title: NLP词向量篇（七）RoBERTa
date: 2020-09-22 15:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（七）RoBERTa"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# RoBERTa: A Robustly Optimized BERT Pretraining Approach
### 1. Paper Information

> 时间：2019年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://arxiv.org/pdf/1907.11692.pdf
>
> 引用：Liu Y, Ott M, Goyal N, et al. Roberta: A robustly optimized bert pretraining approach[J]. arXiv preprint arXiv:1907.11692, 2019.

### 2. Motivation

 &emsp;&emsp; 目前，预训练语言模型已经取得了很好的成果，但是很难去比较不同的预训练模型的效果。而且，训练预训练模型代价又很大，有些模型甚至采用了一些私有的不同大小的数据集，因此很难复现。所以，我们就很难去了解究竟是哪些因素使我们得到了这么好的效果。



### 3. Main Arguments

 &emsp;&emsp; **这篇论文就对BERT的预训练模型进行了研究，比较了在不同的超参数下和不同训练数据集大小的情况下的效果**。通过比较，我们发现，BERT的训练远远没有达到完美，在经过调参后，我们得到了SOTA的水平。我们把这个模型称为RoBERTa。

 &emsp;&emsp; 对BERT的改进主要有以下几点：

- 用更大的batch、更多的数据对模型进行更长时间的训练
- 移除next sentence prediction任务
- 在更长的句子上进行训练
- 动态的改变，对训练数据的mask模式

&emsp;&emsp; 该论文的主要贡献如下：

- 提出了一套BERT的超参选择和训练策略，使得BERT的性能提升
- 使用了一个新的数据集，CC-NEWS，证明了使用更多的数据集来进行预训练可以进一步提升下游任务的效果
- 进一步的确定了BERT的mask语言模型训练目标是非常有用的。



### 4. Framework

 &emsp;&emsp; RoBERTa从Mask方式、模型输入、NSP Loss、batch size、text encoding等几个方式对传统的BERT训练方式进行了修改，具体修改内容和结果如下：

#### 4.1 Static vs Dynamic Masking

 &emsp;&emsp; 原始的Bert采用的时静态Masking，即在处理数据时，随机的选取15%的词，进行Mask，这样的话，在每个epoch，我们就会遇到同样的训练样本。为了更有效的增加数据多样性，本文采取了动态Masking，即在将数据送入网络前进行Masking，这样每个epoch所遇到的训练样本都会不同，两个训练策略的效果如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/1.png?raw=true" alt="1" style="zoom: 50%;" />

 &emsp;&emsp; 可以看到，动态Masking还是起到了一定的效果。

#### 4.2 Model Input Format And Next Sentence Prediction Loss
 &emsp;&emsp; 我们知道，BERT有两个任务，一个是Mask LM任务，一个是NSP任务，两个任务联合训练，输入要么是带有Mask的句子，要么就是句子对，来预训练得到BERT。在BERT之前的文章中也提到NSP是必要的，但是，今天再重新看一下。

 &emsp;&emsp; 为了做这个实验，作者重新定义了模型的输入，分为四种：

1. SEGMENT-PAIR+NSP，与原始BERT相同
2. SENTENCE-PAIR+NSP，使用句子代替原始BERT中的段落
3. FULL-SENTENCES，使用一个连续的长句子，可能会跨文章，如果跨文章的话，在分界处加[SEP]，但不使用NSP Loss来训练，只用了Mask LM目标函数
4. DOC-SENTENCES，与上面几乎相同，但是长句子不跨文章，一个输入中的句子只来源于一个文章。

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/2.png?raw=true" alt="2" style="zoom: 50%;" />

 &emsp;&emsp; 可以看到，**移除了NSP任务后，效果变好了**。同时，我们发现，将输入改变成这样的格式后，比原始BERT的效果提升很大。但是DOC-SENTENCES方式会导致batch size大小的不确定性，因此之后的实验采用了FULL-SENTENCES。

#### 4.3 Training with large batches

 &emsp;&emsp; 作者任务，在同样的训练代价下（step * batch），batch大一些，效果会更好，其实验结果如下

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/3.png?raw=true" alt="3" style="zoom:50%;" />

#### 4.4 Text Encoding

 &emsp;&emsp; 使用了Byte级别的编码方式，即对于词apple的编码来说，设定编码超参$\ n = 3$ ，那么apple对应的编码为，以每个字符作为中心：

```
 “<ap”, “app”, “ppl”, “ple”, “le>”
```

 &emsp;&emsp; 其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“apple”这个单词，进一步，我们可以用这5个trigram的向量叠加来表示“apple”的词向量。



### 5. Result

 &emsp;&emsp; 除了上面讲到的影响因素，在实验中，作者还对数据集大小以及epoch的大小进行了实验。在XLNet上，数据集比原先大10倍，同时batch size比原先大8倍。

 &emsp;&emsp; 当数据大小一样时，我们在GLUE和SQuAD任务上都有提升。当使用附加的数据时，我们在GLUE排行榜上达到了88.5，而当时SOTA只有88.4。我们的模型在GLUE的4/9个任务（MNLI、QNLI、RTE、STS-B）上实现了SOTA，同时在SQuAD和RACE任务上达到了SOTA水平。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/4.png?raw=true)

#### 5.1 GLUE

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/5.png?raw=true)

#### 5.2 SQuAD

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/6.png?raw=true" alt="6" style="zoom:50%;" />

#### 5.3 RACE

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%83%EF%BC%89RoBERTa/7.png?raw=true" alt="7" style="zoom:67%;" />



### 6. Argument

  &emsp;&emsp; 这篇论文算是BERT的预训练策略，调参策略，将BERT调至最佳，使得BERT在各种任务上又提升了几个点，但是创新性不大。



### 7. Further research

