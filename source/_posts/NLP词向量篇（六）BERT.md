---
title: NLP词向量篇（六）BERT
date: 2020-09-22 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（六）BERT"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> 时间：2018年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://arxiv.org/pdf/1810.04805.pdf
>
> 引用：Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

**摘要：**我们引入了一种新的语言表示模型**BERT，即Bidirectional Encoder Representations from Transformers.** 。不同于最近的语言表示模型(Peters et al., 2018a; Radford et al., 2018)，  BERT被设计用于预训练未标记文本的深层双向表示，通过使用所有层中的左右的上下文来获得。因此，只需一个额外的输出层就可以对预先训练好的BERT模型进行微调，从而为广泛的任务创建SOTA模型，比如问题回答和语言推理，而无需对具体任务的架构进行实质性修改。从概念上说，BERT是简单的，但是，从经验上讲，BERT是强大的。**它在11个NLP任务中获得了SOTA的结果**，包括，将GLUE分数提高到80.5%（获得了7.7%的分数提高），将MultiNLI准确率提高到86.7%（4.6%的提升），将SQuAD v1.1问题回答Test F1提高到93.2（1.5个点的提升），将SQuAD v2.0问题回答Test F1提高到83.1（5.1个点的提升）。

**索引**- 自然语言处理，动态词向量

## 内容分析

 &emsp;&emsp; ELMo是一种feature-based方法，即将token输入ELMo，将得到的representation作为附加的输入特征，进行训练。GPT是一种fine-tuning的方法，但是只是采用了单向的Transformer。很多东西限制了ELMo和GPT的表现。BERT吸取了教训，使用了fine-tuning的训练策略，并借助两个独特的任务使得BERT内的Transformer可以获得双向的上下文，同时，BERT也像GPT那样，对不同任务的输入进行了格式转换，最后使得BERT在11项任务中取得了SOTA的效果。下面，我们来逐渐的对BERT的结构以及使用过程进行解析。

### 1）BERT的架构

 &emsp;&emsp; BERT是什么样的呢？其实很简单，就是很多个Transformer的encoder的堆叠，这就是BERT。**输入为一个序列的token，输出为每个token对应的representation的序列**。

 &emsp;&emsp; 在本文中，作者主要使用了两种参数设置，$\ \text{BERT}_{BASE}(L=12,H=768,A=12,\text{Total Parameters = 110M})$  ，以及$\ \text{BERT}_{LARGE}(L=24,H=1024,A=16,\text{Total Parameters = 340M})$ ，其中$\ L$ 表示Transformer的encoder的层数，$\ H$ 表示隐藏层的神经元，$\ A$ 表示multi-head中head的数目。

### 2）BERT的使用

 &emsp;&emsp; 在了解BERT的细节之前，我们先讲一下BERT是如何工作的？作为一个fine-tuning类型的动态词向量模型。BERT首先在一个大数据集上进行预训练，然后在某个具体的任务上及逆行fine-tuning，即将任务的输入送给BERT，得到输出representation，经过一个线性层，的到最后的输出。在fint-tuning阶段要对BERT的参数进行更新。这就是BERT的使用。

### 3）BERT如何应对不同的下游任务？

#### 3.1） BERT的输入转换

 &emsp;&emsp; 为了实现跨任务使用，而在跨任务时，只需要很小的修改模型架构，GPT和BERT都采用了输入转换。即将不同任务的输入转换成一个序列。比如，文本分类中的输入为一个句子，而文本相似性度量的输入却有两个句子，为了整合不同模型的需要，我们要将其转换成同样的格式，即：将不同的句子concat在一起，中间使用分隔符分开。这就实现了输入转换。

 &emsp;&emsp; 在BERT中，在输入转换部分，在一个序列的头部，加入了[CLS]符号，用做某些任务，下面会对其进行讲解。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/1.png?raw=true)

#### 3.2）BERT如何实现不同的输出

 &emsp;&emsp; 记我们的输入序列为$\ E$ ，其中，头部的[CLS]符号对应的隐藏元的输出为$\ C$ ，第$\ i$ 个token对应的输出为$\ T_i$ 。

**GLUE **

 &emsp;&emsp; 对于文本分类任务，给定一个输入句子，输出文本的类别。如何用BERT实现呢？输入还是这个句子，然后我们得到了BERT的输出，我们使用对应于第一个输入标记([CLS])的最终隐藏向量$\ C \in \mathbb{R}^H$ 作为聚合的representation。fine-tuning期间引入的唯一一个新的参数是分类层权重$\ W  \in \mathbb{R}_{K×H}$ ，其中$\ K$ 是标签的数量。我们用$\ C$ 和$\ W$ 计算标准分类损失，即$\ \log(\text{softmax(CW^T)})$ 。

**SQuAD **

 &emsp;&emsp; Stanford Question Answering Dataset(SQuAD v1.1) 是100k个crowdsourced   问题/答案对的集合，这个任务是什么意思呢？给定一个问题，和一个包含答案的文本，我们要找出答案的区间，即开始位置和结束位置。

 &emsp;&emsp; 因为，这里，就有两个文本的输入了，因此，我们把这两个文本连起来，使用分隔符分开，如图1所示。

 &emsp;&emsp; 在fine-tuning过程中，我们又引入一个起始向量$\ S\in \mathbb{R}^H$ 和一个结束向量$\ E\in \mathbb{R}^H$ 。单词$\ i$ 作为答案区间范围的开头的概率，通过计算为$\ T_i$ 与$\ S$ 之间的点积，然后在所有的词上进行softmax运算得到，即：$\ P_i = \frac{e^{S·T_i}}{\sum_j e^{S·T_i}}$ 。使用类似的公式来计算答案区间的末尾。从位置$\ i$ 到位置$\ j$ 的候选区间的得分被定义为$\ S·T_i+ E· T_j$ ，分数最大的区间被用来作为预测值，同时，要保证$\ j ≥  i$ 。这次的训练目标是正确起始位置和结束位置的对数似然性之和。

### 4）BERT的精华

#### 4.1）BERT如何实现双向Transformer？

 &emsp;&emsp; 截止到现在，BERT跟GPT简直一模一样，但是，开头我们说了，BERT的一个特别大的贡献就是通过两个任务实现了双向Transformer。

##### **Task #1: Masked LM**

 &emsp;&emsp; 为了训练深度双向representation，**我们简单地随机mask一定比例的输入token，然后预测那些被mask的token**。**我们称这个过程为“masked LM”(MLM)**。在这种情况下，对应于mask token的最终隐藏向量被送到softmax中，基于整个词汇表进行分类，就像是一个标准的LM。在我们所有的实验中，我们随机mask每个序列中所有单词块token的15%。与去噪自动编码器不同，我们只预测被mask的词，而不是重建整个输入。

 &emsp;&emsp; 虽然这允许我们获得一个双向的预训练模型，但缺点是我们在预训练和fine-tuning之间造成了不匹配，因为[MASK]标记在fine-tuning期间不会出现。为了缓解这种情况，**我们并不总是用实际的[MASK]标记 来替换被mask的单词。训练数据生成器会随机选择15%的token位置进行预测。如果选择了第$\ i$ 个令牌，我们将第$\ i$ 个令牌替换为:(1)80%的时间使用[MASK]token，(2)10%的时间使用随机token，(3)10%的时间使用未更改的第$\ i$ 个token。然后，将使用$\ T_i$ 和交叉熵损失被用来预测原始token**。我们在附录C.2中比较了该步骤的变化。; 

##### Task #2:  Next Sentence Prediction (NSP)

 &emsp;&emsp; 许多重要的下游任务，如**问答(QA)和自然语言推理(NLI)都是基于理解两个句子之间的关系**，这不是语言建模直接捕捉的。**为了训练一个可以理解句子关系的模型，我们对二分类的下一句预测任务（next sentence prediction ）进行预训练**，该任务可以从任何单语语料库中简单地生成。具体来说，当选择句子A和B作为预训练的样本时，50%的时间B是跟随在A后面的实际的下一个句子(标签为$\ IsNext$ )，50%的时间它是来自语料库的随机句子(标签为$\ NotNext$ )。如图1所示，$\ C$ 用于下一句预测(NSP)，即利用$\ C$ 来预测标签，$\ isNext\ or\ NotNext$ 。尽管它很简单，但我们在第5.1节中证明了对这项任务的预训练，对质量保证和NLI都非常有益。NSP任务与 Jernite et al. (2017) and Logeswaran and Lee (2018)使用的表征学习目标密切相关。然而，在以前的工作中，只有sentence词向量被迁移到下游任务，而BERT通过迁移所有参数来初始化终端任务模型参数。

#### 4.2）BERT的输入只有token词向量吗？

 &emsp;&emsp; **对于给定的token，其输入representaion是通过对相应的token、segment和position embeddings求和来构建的**。这种结构的可视化可以在图2中看到。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/2.png?raw=true)



### 5）BERT的效果

 &emsp;&emsp; BERT在11个NLP任务上取得了SOTA的效果，具体的实验效果可以看第四节。之后作者基于BERT的模型大小、BERT的各部分进行了ablation的研究，实验结果看第五节。而且，作者还将BERT改造成了feature-base的模型，观察了一下结果。

## 1、Introduction

 &emsp;&emsp; 语言模型预训练已经被证明对许多自然语言处理任务是有效的(Dai and Le, 2015; Peters et al., 2018a; Radford et al., 2018; Howard and Ruder,  2018)。其中包括sentence级任务，如自然语言推理（natural language processing）(Bowman et al., 2015; Williams et al., 2018)和释义（paraphrasing）(多兰和Brockett,  2005)，旨在通过从整体分析句子之间的关系预测这些句子，以及token级任务，比如命名实体识别（named entity recognition）和回答问题（question answering），其中，模型需要去生成token级别的输出(TTjong Kim Sang and De Meulder, 2003; Rajpurkar et al., 2016)。

 &emsp;&emsp; **在下游任务中应用预训练的语言表示的现有策略有两种：feature-based方法和fine-tuning方法。feature-based方法，如ELMo** (Peters et al., 2018a)，在下游任务，使用特定于任务的架构，其中包括预训练过的representation作为附加特征。**fine-tuning方法，如生成式预训练的Transformer (OpenAI GPT)**  (Radford et al., 2018)，引入了最小的特定于任务的参数，并通过简单地微调所有预训练参数来对下游任务进行训练。**这两种方法在预训练过程中使用了相同的目标函数，即使用单向语言模型来学习一般语言表征**。

 &emsp;&emsp; 我们认为，**目前的技术限制了预训练的representation的能力，特别是fine-tunning方法。主要的限制是标准语言模型是单向的，这限制了在训练前可以使用的架构的选择**。例如，在inOpenAIGPT中，作者采用了一种从左到右的架构，其中每个token只能关注transformer的self-attention层中的前一个token(V aswani et al.,2017)。这样的限制对于sentence级的任务来说，得不到最优的模型，而且在应用基于fine-tunning的方法来完成诸如问题回答这样的token级任务时可能非常有害，在这些任务中，从两个方向结合上下文是至关重要的。

 &emsp;&emsp; 在本文中，我们改进了基于fine-tuning的方法，提出了BERT：Encoder Representations from Transformers。BERT受完形填充（Cloze）任务的启发，通过**使用“masked language model”(MLM)的预训练目标缓解了上述单向约束**(Taylor,  1953)。masked language model随机mask输入中的一些token，其目标是仅根据上下文预测掩码词的原始词汇表id。与从左到右语言模型的预训练不同，MLM目标使representation能够融合左右上下文，这使我们能够预训练一个深度的双向转换器。除了掩蔽语言模型之外，我们还使用了一个“next sentence prediction”任务，它联合地预训练了text-pair 表示。本文的贡献如下：

- 我们**论证了双向预训练对语言表征的重要性**。与Radford等人(2018)使用单向语言模型进行预训练不同，**BERT使用掩码语言模型来实现预训练的深度双向表示**。这也与Peters等人(2018a)形成了对比，Peters等人使用独立训练的从左到右和从右到左的LMs的浅连接。
- 我们表明，预先训练的representation减少了对许多精心设计的特定于任务的架构的需求。BERT是第一个基于fine-tuning的表示模型，它在大量sentence级和token级任务上实现了SOTA性能，优于许多特定任务的架构。
- BERT提高了11个NLP任务的技术水平。代码和预训练的模型可以在https://github.com/google-research/bert获得。



## 2、Related Work

 &emsp;&emsp; 预训练的语言表示有很长的历史了，我们在这一节简要的回顾了一下最常用的方法。

### 2.1 Unsupervised Feature-based Approaches

 &emsp;&emsp; 学习广泛适用的词向量是近十年来研究的一个活跃的领域，包括non-neural方法（Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006）和neural方法（Mikolov et al., 2013; Pennington et al., 2014）预训练过的词向量是现代NLP系统的一个组成部分，相对于从零开始学习词向量，预训练的方法提供了显著的改进(Turian et al., 2010)。为了预训练词向量，可以使用从左到右的语言建模目标(Mnih and Hinton, 2009)，以及在左右语境中区分正确和不正确单词的目标(Mikolov et al., 2013)。

 &emsp;&emsp; 这些方法已经被推广到更粗的粒度，如sentence embeddings (Kiros et al., 2015; Logeswaran and Lee, 2018) 或paragraph embeddings (Le and Mikolov, 2014)。为了训练sentence representations，之前的工作使用对候选的下一个句子进行排名的目标(Jernite et al., 2017; Logeswaran and Lee, 2018)，即，给出前一句的representation，从左到右生成下一句单词(Kiros et al., 2015)，或去噪自编码器派生的目标(Hill et al., 2016)。

 &emsp;&emsp; ELMo和它的前身(Peters et al., 2017, 2018a)从不同的维度概括了传统的词嵌入研究。它们从从左到右和从右到左的语言模型中提取上下文敏感的特性。每个token的上下文表示是从左到右和从右到左表示的串联。当将上下文词向量集成到现有的特定任务架构时，ELMo提高了几个主要的NLP基准(Peters et al., 2018a)的技术水平，包括问题回答(Rajpurkar et al., 2016)、情绪分析(Socher et al., 2013)和命名实体识别(Tjong Kim Sang and De Meulder, 2003)。Melamud等人(2016)提出通过使用LSTMs从左右语境中预测单个单词来学习语境表征。与ELMo相似，他们的模型是feature-based 的，不是深度双向的。Fedus等人(2018)表明，完形填空任务可以用来提高文本生成模型的鲁棒性。

### 2.2 Unsupervised Fine-tuning Approaches

 &emsp;&emsp; 与feature-based的方法一样，第一种在这个方向上的方法只能从未标记的文本中提取预训练好的词向量参数(Collobert和Weston, 2008)。

 &emsp;&emsp; 最近，生成上下文token representation的sentence或document编码器已经从未标记的文本中进行了预训练，并在一个监督下游任务进行fine-tuning(Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018)。这些方法的优点是几乎不需要从头学习参数。利用这一优势，OpenAI GPT  (Radford et al., 2018)在GLUE基准测试的许多句子水平任务上取得了之前SOTA的结果(Wang et al., 2018a)。从左到右的语言建模和自动编码器目标，已经被使用在了许多与训练任务中，例如(Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015)。

### 2.3 Transfer Learning from Supervised Data

 &emsp;&emsp; 也有研究显示了从大型数据集的监督任务进行有效的迁移学习，如自然语言推理(Conneau et al., 2017)和机器翻译(McCann et al., 2017)。计算机视觉研究也证明了从大型预训练模型中进行迁移学习的重要性，有效的方法是对用ImageNet预训练的模型进行微调(Deng et al., 2009; Y osinski et al., 2014)。

## 3、BERT

 &emsp;&emsp; 我们将在本节中介绍BERT及其详细实现。在我们的框架中有**两个步骤：pre-training 和 fine-tuning**。在预训练过程中，对不同的预训练任务中的未标记数据进行训练。在fine-tunning阶段，首先使用预先训练好的参数初始化BERT模型，然后**使用来自下游任务的标记数据对所有参数进行微调**。每个下游任务都有单独的经过优化的模型，即使它们使用相同的预训练参数初始化。图1中的问题回答示例将作为本节的运行示例。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/1.png?raw=true)

 &emsp;&emsp; **BERT的一个显著特征是它跨不同任务的统一架构。预训练的架构和最终的下游架构之间的差别很小。**

**Model Architecture**

 &emsp;&emsp; BERT的模型架构是基于Vaswani et al. (2017)所描述的多层双向Transformer编码器的原始架构实现的，其在tensor2tensor 库中已经发布。由于Transformer的使用已经非常普遍，而且我们的实现几乎与最初的实现完全相同，因此我们将省略对模型架构的详尽背景描述，并请读者参考Vaswani et al. (2017)以及优秀的指南，如“The Annotated Transformer”。

 &emsp;&emsp; 在本文中，我们记 Transformer block的层数为$\ L$ ，隐藏层大小为$\ H$ ，self-attition head的数目记为$\ A$ 。**我们主要采用了两个模型大小：$\ \text{BERT}_{BASE}(L=12,H=768,A=12,\text{Total Parameters = 110M})$  ，以及$\ \text{BERT}_{LARGE}(L=24,H=1024,A=16,\text{Total Parameters = 340M})$**  。

 &emsp;&emsp; 处于比较的目的，$\ \text{BERT}_{BASE}$  跟OpenAI GPT具有相同的大小。关键的是，BERT的Transformre使用了双向self-attention，但是GPT的Transformer使用了受限的self-attention，每一个token只能注意到它上文。（**双向Transformer就是Transformer的encoder，受限的Transformer，即只是用左侧语境的Transformer就是Transformer decoder**）

**Input/Output Representations**

 &emsp;&emsp; 为了让BERT能够处理各种下游任务，我们的输入representation能够在一个token序列中明确地表示一个句子或一对句子(例如，$\ <Question,Answer>$ )。在本文中，“sequence”是连续文本的任意范围，而不是实际的语言句子。一个“sequence”是指输入给BERT的token序列，可以是单个句子，也可以是打包在一起的两个句子。

 &emsp;&emsp; 我们使用WordPiece词向量(Wu et al., 2016)，具有30000个token词汇表。**每个sequence的第一个token总是一个特殊的分类标记($\ [\text{CLS}]$ )。对应于该token的最终隐藏状态被用作分类任务的聚合序列表示。**句子对 被打包成一个单一的序列。**我们用两种方法区分句子。首先，我们用一个特殊的token($\ [\text{SEP}]$ )将它们分开。其次，我们为每个token添加一个可学习的词向量，指示它是属于句子A还是句子B。**如图1所示，我们将输入词向量表示为E，特殊$\ [\text{CLS}]$ 标记的最终隐藏向量表示为$\ C \in \mathbb{R}^H$ ，输入token的最终隐藏向量表示为$\ T_i \in \mathbb{R}^H$ 。

 &emsp;&emsp; **对于给定的token，其输入representaion是通过对相应的token、segment和position embeddings求和来构建的**。这种结构的可视化可以在图2中看到。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/2.png?raw=true)

### 3.1 Pre-training BERT

 &emsp;&emsp; 与Peters et al. (2018a) and Radford et al. (2018)不同，我们不使用传统的从左到右或从右到左的语言模型来预训练BERT。相反，我们使用两个无监督任务对BERT进行预训练，如本节所述。这一步显示在图1的左侧。

#### 3.1.1 Task #1: Masked LM

 &emsp;&emsp; 直觉上，有理由相信深度双向模型比从左到右模型或从左到右和从右到左模型的浅层concat连接更强大。不幸的是，标准条件语言模型只能从左到右或从右到左训练，因为双向条件需要允许每个单词间接“看到自己”，并且该模型可以在多层上下文中简单地预测目标单词。

 &emsp;&emsp; 为了训练深度双向representation，**我们简单地随机mask一定比例的输入token，然后预测那些被屏mask的token**。**我们称这个过程为“masked LM”(MLM)**，尽管在文献中它经常被称为完形填空任务(Taylor, 1953)。在这种情况下，对应于mask token的最终隐藏向量被送到softmax中，对整个词汇表进行分类，就像是一个标准的LM。在我们所有的实验中，我们随机mask每个序列中所有单词块token的15%。与去噪自动编码器(Vincent et al., 2008)相反，我们只预测被mask的词，而不是重建整个输入。

 &emsp;&emsp; 虽然这允许我们获得一个双向的预训练模型，但缺点是我们在预训练和fine-tuning之间造成了不匹配，因为[MASK]标记在fine-tuning期间不会出现。为了缓解这种情况，**我们并不总是用实际的[MASK]标记 来替换被mask的单词。训练数据生成器会随机选择15%的token位置进行预测。如果选择了第$\ i$ 个令牌，我们将第$\ i$ 个令牌替换为:(1)80%的时间使用[MASK]token，(2)10%的时间使用随机token，(3)10%的时间使用未更改的第$\ i$ 个token。然后，将使用$\ T_i$ 和交叉熵损失被用来预测原始token**。我们在附录C.2中比较了该步骤的变化。

#### 3.1.2 Task #2:  Next Sentence Prediction (NSP)

 &emsp;&emsp; 许多重要的下游任务，如**问答(QA)和自然语言推理(NLI)都是基于理解两个句子之间的关系**，这不是语言建模直接捕捉的。**为了训练一个可以理解句子关系的模型，我们对二分类的下一句预测任务（next sentence prediction ）进行预训练**，该任务可以从任何单语语料库中简单地生成。具体来说，当选择句子A和B作为预训练的样本时，50%的时间B是跟随在A后面的实际的下一个句子(标签为$\ IsNext$ )，50%的时间它是来自语料库的随机句子(标签为$\ NotNext$ )。如图1所示，$\ C$ 用于下一句预测(NSP)，即利用$\ C$ 来预测标签，$\ isNext\ or\ NotNext$ 。尽管它很简单，但我们在第5.1节中证明了对这项任务的预训练，对质量保证和NLI都非常有益。NSP任务与 Jernite et al. (2017) and Logeswaran and Lee (2018)使用的表征学习目标密切相关。然而，在以前的工作中，只有sentence词向量被迁移到下游任务，而BERT通过迁移所有参数来初始化终端任务模型参数。

#### 3.1.3 Pre-training data

 &emsp;&emsp; 预训练程序在很大程度上遵循现有的语言模型预训练文献。对于预训练语料库，我们使用了BooksCorpus(800M单词)(Zhu et al., 2015)和English Wikipedia(2,500M words)。对于维基百科，我们只提取文本段落，忽略列表、表格和标题。为了提取长的连续序列，**使用document级语料库而不是诸如Billion Word Benchmark (Chelba et al., 2013)的混合sentence级语料库是至关重要的**。

### 3.2 Fine-tuning BERT

 &emsp;&emsp; fine-tunning很简单，因为Transformer中的self-attention机制允许BERT通过交换适当的输入和输出来模拟许多下游任务，无论它们涉及单个文本还是文本对。对于涉及文本对的应用，一种常见的模式是在应用bidirectional cross attention之前对文本对进行独立编码，如Parikh et al. (2016); Seo et al. (2017)。相反，BERT使用self-attention机制来统一这两个阶段，因为编码具有self-attention信息的concat后的文本对，能够有效地包含两个句子之间的bidirectional cross attention。

 &emsp;&emsp; **对于每个任务，我们只需将特定于任务的输入和输出插入BERT，并端到端地微调所有参数。**在输入端，来自预训练的句子A和句子B类似于(1)释义中的句子对，(2)文本蕴涵中的假设-前提对，(3)问题回答中的问题-段落对，以及(4)文本分类或序列标记中的退化text-∅对。**在输出端，token representation被送到用于token级任务的输出层，例如序列标记或问题回答，而[CLS] representation被送到用于分类的输出层，例如蕴涵或情感分析。**

 &emsp;&emsp; 相比于预训练，fine-tuning的计算代价相对小一些。从完全相同的预训练模型开始，论文中的所有结果（fine-tuning阶段）最多可以在单个云TPU上运行1小时，或者在GPU上运行几个小时。我们将在第4节的相应小节中描述特定任务的详细信息。更多详情见附录A.5。

## 4、Experiments

 &emsp;&emsp; 在本节中，我们将介绍11个NLP任务的BERT fine-tuning结果。

### 4.1 GLUE

 &emsp;&emsp; General Language Understanding Evaluation (GLUE) benchmark(Wang et al., 2018a)是多种自然语言理解任务的集合。附录B.1中包含了GLUE数据集的详细描述。

 &emsp;&emsp; 为了在GLUE上进行fine-tuning，我们按照第3节所述表示输入序列(对于单句或句子对)，**并使用对应于第一个输入标记([CLS])的最终隐藏向量$\ C \in \mathbb{R}^H$ 作为聚合的representation。fine-tuning期间引入的唯一一个新的参数是分类层权重$\ W  \in \mathbb{R}_{K×H}$ ，其中$\ K$ 是标签的数量。我们用$\ C$ 和$\ W$ 计算标准分类损失，即$\ \log(\text{softmax(CW^T)})$ 。**

 &emsp;&emsp; 我们使用batch=32，并对所有GLUE任务的数据进行3个epoch的fine-tuning。对于每个任务，我们在Dev集上选择了最佳fine-tuning学习率(在5e-5、4e-5、3e-5和2e-5中)。此外，对于$\ \text{BERT}_{LARGE}$ ，我们发现有时在小数据集上进行fine-tuning是不稳定，因此我们运行了几次随机重启，并在Dev集上选择了最佳模型。通过随机重启，我们使用相同的预训练检查点，但执行不同的fine-tuning数据随机和分类器层初始化。

 &emsp;&emsp; 结果如表1所示。在所有任务上，$\ \text{BERT}_{BASE}$ 和$\ \text{BERT}_{LARGE}$ 在所有任务上，相比于之前的SOTA，性能都远远超过，它们的平均准确率分别提高了4.5%和7.0%。请注意，$\ \text{BERT}_{BASE}$和OpenAI  GPT在模型架构方面几乎完全相同，只是GPT中的attention使用了masking。对于最大和最广泛报道的GLUE任务，MNLI，BERT获得了4.6%的绝对精度提高。在官方排名榜中，$\ \text{BERT}_{LARGE}$ 获得了80.5分，相比之下，在本报告撰写之日，GPT公开赛只获得了72.8分。

 &emsp;&emsp; 同时，我们发现，$\ \text{BERT}_{LARGE}$ 在所有任务上都远远超过$\ \text{BERT}_{BASE}$ ，特别是数据量比较少的几个数据集。模型大小的影响在5.2节中介绍。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/3.png?raw=true)

### 4.2 SQuAD v1.1

 &emsp;&emsp; Stanford Question Answering Dataset(SQuAD v1.1) 是100k个crowdsourced   问题/答案对的集合(Rajpurkar et al., 2016)。给定一个问题和维基百科中包含答案的一段话，任务是预测这段话中的答案文本的区间范围。

 &emsp;&emsp; 如图1所示，在问答任务中，我们将输入的问题和段落表示为一个打包的序列，其中问题使用A词向量，段落使用B词向量。**在fine-tuning过程中，我们又引入一个起始向量$\ S\in \mathbb{R}^H$ 和一个结束向量$\ E\in \mathbb{R}^H$ 。单词$\ i$ 作为答案区间范围的开头的概率，通过计算为$\ T_i$ 与$\ S$ 之间的点积，然后在所有的词上进行softmax运算得到，即：$\ P_i = \frac{e^{S·T_i}}{\sum_j e^{S·T_i}}$ 。使用类似的公式来计算答案区间的末尾。从位置$\ i$ 到位置$\ j$ 的候选区间的得分被定义为$\ S·T_i+ E· T_j$ ，分数最大的区间被用来作为预测值，同时，要保证$\ j ≥  i$** 。这次的训练目标是正确起始位置和结束位置的对数似然性之和。我们fine-tuning了3个epoch，学习率为5e-5，batch为32。

 &emsp;&emsp; 表2显示了排名榜中top系统的结果以及目前最好地已发布的系统的结果(Seo等人，2017；克拉克和加德纳，2018；Peters等人，2018a胡等，2018)。SQuAD排行榜的前几名并没有发布其信息，并没有说有没有用其他的公用数据集信息。因此，我们在我们的系统中使用适度的数据增强，首先在TriviaQA数据集上进行fine-tuning(Joshi et al., 2017)，然后在SQuAD上进行微调。

 &emsp;&emsp; 我们的最好的系统，通过集成的策略，比排行榜上最好的系统高出1.5个F1分数点，如果是单个模型，则高出1.3个点。事实上，我们的BERT模型要比前几名的集成模型的性能要好。而，如果没有TriviaQA作为预训练集，我们仅仅损失0.1-0.4个F1分数点，仍然要比他们好。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/4.png?raw=true)

### 4.3 SQuAD v2.0

 &emsp;&emsp; SQuAD2.0 任务拓展了SQuAD1.1任务的定义，其允许了问题没有答案的这种可能性，使得问题更加的真实。

 &emsp;&emsp; 我们使用一个简单的方法来为这个任务扩展SQuAD v1.1 的 BERT模型。我们将没有答案的问题，视为其答案在一个以[CLS]符号开始和结束的答案区间。开始和结束答案区间位置的概率空间被扩展，以包括[CLS]token的位置。为了进行预测，我们将无答案区间的得分：$\ s_{null}  = S ·C+E ·C$ 与最佳非空区间的得分$\ \hat{s_{i,j}} = \max_{j \ge i} S·T_i + E·T_j$ 进行比较，当$\ S_{i,j} >  s_{null}+ \tau$ 时，在dev集上选择阈值$\ τ$ 以最大化F1。我们没有在这个模型中使用TriviaQA数据。我们fine-tuning了2个epoch，学习率为5e-5，batch为48。

 &emsp;&emsp; 在表三中展示了，与先前的排行榜的模型和发表过的最优秀的几个模型 (Sun et al., 2018; Wang et al., 2018b)的对比，我们获得了5.1个F1分数的提升。

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/5.png?raw=true)

### 4.4 SWAG

 &emsp;&emsp; Situations With Adversarial Generations（SWQG）包含113k个句子对完成例子，用于评估基础常识推理(Zellers et al., 2018)。给定一个句子，任务是在四个选项中选择最合理的续集。

 &emsp;&emsp; 在对SWAG数据集进行fine-tuning时，我们构建了四个输入序列，每个序列包含给定句子(句子A)和可能的延续(句子B)的concat。引入的唯一特定于任务的参数是一个向量，其与[CLS]token representation的点积表示每个选项的分数，该分数用softmax层进行规范化。

 &emsp;&emsp; 我们fine-tuning模型3个epoch，学习率为2e-5，batch为16。在表4中呈现了结果，$\ \text{BERT}_{LARGE}$ 的效果要比作者的baeline ESIM + ELMo的效果好27.1%，比GPT好8.3%。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/6.png?raw=true)

## 5、Ablation Studies

 &emsp;&emsp; 在这一节，我们进行了ablation实验，为乐趣更好的理解BERT各部分的相对重要性，附加的ablation研究在附录C中。

### 5.1 Effect of Pre-training Tasks

 &emsp;&emsp; 通过评估两个预训练目标（使用完全一样的与训练数据、fine-tuing策略、与$\ \text{BERT}_{BASE}$ 一样的超参），我们证明了BERT的深度双向性的重要性。

**No NSP** ：一个使用MLM训练的双向模型，但没有使用NSP任务进行训练。

**LTR & No NSP** ：一个使用标准的 Left-to-Right (LTR) 语言模型训练的单向模型（只考虑上文），而并没有使用MLM。在fine-tuning过程中也保持了单向的限制，因为用双向的话会导致预训练和fine-tuing阶段的不匹配，这会是下游任务的性能下降。另外，这个模型没有用NSP任务进行训练，这一个是跟GPT差不多的，但是使用了我们的更大的训练数据集、我们的输入representation和我们的fine-tuning策略。

 &emsp;&emsp; 我们首先考察NSP任务带来的影响。在表5中，我们显示移除NSP会显著影响在QNLI、MNLI和 SQuAD 1.1数据集上的表现。接下来，我们通过比较“No NSP”和“LTR & No NSP”来评估训练双向表征的影响。LTR模型在所有任务上的表现都比MLM模型差，在MRPC和SQuAD上有很大的下降。

 &emsp;&emsp; 对于SQuAD来说，很明显LTR模型在token预测方面表现不佳，因为token级别的隐藏状态没有下文。为了真诚地尝试加强LTR系统，我们在顶部添加了一个随机初始化的BiLSTM。这确实显著地改善了在SQuAD上的结果，但是结果仍然比预先训练的双向模型差得多。BiLSTM损害了GLUE任务的性能。

 &emsp;&emsp; 我们认识到，也可以训练单独的LTR和RTL模型，并将每个token表示为两个模型的串联，就像ELMo一样。但是:(a)这比单个双向模型训练代价大一倍；（b）对于QA这样的任务来说，这是 non-intuitive的，因为RTL模型不能将问题的答案作为条件；(c) 这严格来说不如深度双向模型强大，因为它可以在每一层都使用左右上下文。

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/7.png?raw=true)

### 5.2 Effect of Model Size

 &emsp;&emsp; 在本节中，我们探讨模型大小对微调任务准确性的影响。我们用不同数量的层、隐藏单元和注意力头训练了许多的BERT模型，同时使用了与前面描述的相同的超参数和训练程序。

![8](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%85%AD%EF%BC%89BERT/8.png?raw=true)

### 5.3 Feature-based Approach with BERT

 &emsp;&emsp; 到目前为止，所有BERT结果都使用了fine-tuning方法，即在预训练的模型中添加一个简单的分类层，并在下游任务中联合fine-tuning所有参数。然而，基于特征的方法(从预处理模型中提取固定特征)具有一定的优势。首先，并不是所有的任务都可以很容易地用Transformer编码器架构来表示，因此需要添加特定于任务的模型架构。第二，预先计算一次昂贵的训练数据representation，然后在该representation的基础上用更便宜的模型运行许多实验，这有很大的计算优势。

 &emsp;&emsp; 在这一节中，我们比较了两种方法，通过将BERT应用到NER任务中，进行比较。BERT的输入，即词向量部分，我们使用了区分大小写的WordPiece model，并且我们包含了数据提供的最大文档上下文。按照标准做法，我们将其表述为标记任务，但在输出中不使用CRF层。我们使用第一个子token的representation作为NER标签集上token级分类器的输入。

 &emsp;&emsp; 为了消除fine-tuning方法，我们应用基于特征的方法，从一个或多个层提取activations，而不微调BERT的任何参数。这些上下文embedding被用作分类层之前的随机初始化的两层768维BiLSTM的输入。

 &emsp;&emsp; 在表7中给出了结果，可以看到$\ \text{BERT}_{LARGE}$ 的表现可以与SOTA有一拼。最好地那个模型是将预训练的Transformer最顶部的四层的representation，concat在一起，这种方法只比fine-tuning整个模型少0.3个F1分数点。这也就证明了，对于fine-tuning和基于特征的方法都是有效的。

## 6、Conclusion

 &emsp;&emsp; 最近通过语言模型进行的迁移学习带来的经验改进表明，丰富的、无监督的预训练是许多语言理解系统不可或缺的一部分。特别是，这些结果使得即使是低资源的任务也能从深度单向架构中获益。我们的主要贡献是进一步将这些发现推广到深度双向架构，允许相同的预训练模型成功处理广泛的自然语言处理任务。