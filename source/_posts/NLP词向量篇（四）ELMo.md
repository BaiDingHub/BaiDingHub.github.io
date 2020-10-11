---
title: NLP词向量篇（四）ELMo
date: 2020-09-21 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（四）ELMo"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# Deep contextualized word representations
> 时间：2018年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://arxiv.org/pdf/1802.05365.pdf
>
> 引用：Peters M E, Neumann M, Iyyer M, et al. Deep contextualized word representations[J]. arXiv preprint arXiv:1802.05365, 2018.

**摘要：**我们引入了一种**新型的深层语境化词向量**，它既**模拟了(1)复杂的词汇使用特征(例如，句法和语义)，也模拟了(2)这些用法如何在不同的语言语境中变化(例如，模拟多义词)**。我们的词向量是深双向语言模型(deep bidirectional language model，biLM)内部状态的可学习函数，该模型是在大型文本语料库上预先训练的。我们表明，这些词向量可以很容易地添加到现有的模型中，并在六个具有挑战性的NLP问题(包括问题回答（question answering）、文本蕴涵（textual entailment）和情绪分析（sentiment analysis）)中显著提高技术水平。我们还提供了一项分析，显示出预训练网络的深层内在结构是至关重要的，允许下游模型混合不同类型的半监督信号。

**索引**- 自然语言处理，词向量

## 内容分析

 &emsp;&emsp; 最近词向量大火，Word2vec、Glove等等静态词向量给NLP任务带来了很大的提升，但是我们都知道，不同的数据集中同一个词的表示含义可能是不同的，而且也有可能是该数据集特有的一些含义，虽然静态的词向量可以表现出多义，但是应用在这种情况下可能就显得不太够了，因此，我们需要根据数据集的自身特点来对词向量进行微调，这就是动态词向量，也是ELMo的主要思想，下面我们来看看ELMo的具体结构。

### 1）ELMo的结构

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/1.png?raw=true)

 &emsp;&emsp; ELMo的输入就是一个词的词向量$\ [E_1,...,E_N]$ ，将词向量输入到一个前向LSTM和一个后向LSTM，我们就得到了两个输出值，之后将这两个输出值再送入前向和后向LSTM中，这样重复$\ L=2$ 次，同时，在层与层之间添加了残差链接。在每一层，我们都有两个输出值，再加上最开始的词向量，我们就有了$\ 2L+1$ 个向量，我们将其记为：
$$
\begin{equation}
\begin{split}
R_k &= \{x_k^{LM},\vec{h_{k,j}^{LM}}, \overleftarrow{h_{k,j}^{LM}} | j = 1,...,L\}\\
&={h_{k,j}^{LM} | j = 0,...,L}
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中$\ k$ 表示这是运行的第$\ k$ 个token，$\ $ $\ h_{k,0}^{LM}$ 是token层的词向量，$\ h_{k,j}^{LM} = [\vec{h_{k,j}^{LM}}, \overleftarrow{h_{k,j}^{LM}]}$ 表示每一个biLSTM层（双向LSTM，前向+后向）的输出。

 &emsp;&emsp; 那这$\ 2L+1$ 个向量有什么用呢？我们使用这$\ 2L+1$ 个向量来重新表示词向量，因为biLSTM包含着数据集的信息，因此，经过调整后的representation代表的含义会更有用，我们一般这样来整合这$\ 2L+1$ 个向量：
$$
ELMo_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k,j}^{LM} \tag{1}
$$
 &emsp;&emsp; 其中$\ \gamma$ 是一个标量，用来规范最后得到的representation的大小，$\ s_j$ 是用Softmax生成的归一化因子，用来整合不同层的representation的。这样的话，我们就最后得到了第k个token的representation，$\ ELMo_k^{task}$ ，不过，在使用的时候，我们一般既使用词向量，也是用得到的representation，即$\ [x_k;ELMo_k^{task}]$ 。

### 2）ELMo的使用

 &emsp;&emsp; 我们会在大数据集上预训练ELMo，包含多种任务。之后，我们把ELMo加入到下游任务中，我们用ELMo生成token在每一个biLSTM上的representation，然后让下游任务自己去学习不同层的representation的关系。

### 3）ELMo分析

#### ① 双向biLSTM的使用

 &emsp;&emsp; 双向biLSTM使得模型可以得到token的上下文信息。

#### ② ELMo的小trick

 &emsp;&emsp; 可以在ELMo的biLSTM层后添加dropout。

 &emsp;&emsp; 我们可以将biLSTM的权重进行正则化，即添加正则化因子$\ \lambda||w||_2^2$ 。

#### ③ ELMo的 representation的最终选择的影响

 &emsp;&emsp; 如果，我们只使用了ELMo的最后一层的输出作为该token的representation，那么效果会比使用所有的层的输出 会差一点，结果如四中的图。

#### ④ 正则化因子对效果的影响

 &emsp;&emsp; 经过分析发现，当$\ \lambda = 1$ 时，基本上是对biLSTM各层取平均，当$\ \lambda$ 较小时，ELMo会学习自己的权重分布，这样得到的效果会很好，数据集较小的时候，对$\ \lambda$ 会很敏感

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/3.png?raw=true)

#### ⑤ ELMo添加位置的影响

 &emsp;&emsp; 我们除了可以在词向量输入到模型的时候，使用ELMo进行调整外，我们还可以在下游模型的encoder的输出$\ h_k$ 上添加ELMo的representation，即组成$\ [h_k,ELMo_k^{task}]$ 。我们会发现有部分在输入输出位置都加，效果会好一些，而有一些只在输入位置加，效果会好一些。这是因为，有些任务会更喜欢本模型生成的上下文表示$\ h_k$ ，而不太中意预训练得到的$\ ELMo_k^{task}$ 。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/4.png?raw=true)

#### ⑥ ELMo到底捕获了什么信息？

 &emsp;&emsp; 正如开头所说的，ELMo解决了基于特定数据集而存在的多义词问题。



#### ⑦ ELMo的效果

 &emsp;&emsp; 总结来说，又快又好，添加ELMo之前，我们需要训练486个epoch才能够达到F1分数的最大值，添加ELMo后，模型在10个epoch时超过了baseline的最大值，达到相同性能水平所需的更新数量相对减少了98%。同时，在数据量很小的时候，添加ELMo的表现要比未添加的好很多。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/6.png?raw=true)

 &emsp;&emsp; 在不同的任务上的效果如下：

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/2.png?raw=true)

#### ⑧ 在不同任务中，对不同层的权重偏爱程度？

 &emsp;&emsp; 图2显示了softmax规范化的学习层权重。在输入层，对于coreference和SQuAD来说，任务模型倾向于第一个biLSTM层。但是对于其他任务来说，它的分布并没有达到峰值。输出层的权重相对平衡，较低的层略有偏好。

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/7.png?raw=true)

 

## 1、Introduction

 &emsp;&emsp; 预训练的单词表示(Mikolov et al., 2013; Pennington et al., 2014)是许多神经语言理解模型的关键组件。然而，学习高质量的词向量是具有挑战性的。他们应该很好的模拟(1)复杂的词汇使用特征(例如，句法和语义)，以及(2)这些用法如何在不同的语言语境中变化(例如，模拟多义词)。在本文中，我们引入了一种新型的深度语境化词向量，它可以直接解决这两种挑战，可以轻松地集成到现有的模型中，并且在跨越一系列具有挑战性的语言理解问题的每一种考虑的情况下显著改进技术状态。

 &emsp;&emsp; 我们的词向量方法与传统的词向量方法不同，在传统的词向量方法中，每个token都被分配了一个词向量，而我们的词向量是整个输入句子的一个函数。我们使用来自双向LSTM的向量，该LSTM在大型文本语料库上经过耦合语言模型(LM)目标训练。出于这个原因，**我们称它们为ELMo(Embeddings from Language Models。与以往学习上下文化词向量的方法不同(Peters et al., 2017; McCann et al., 2017，ELMo的词向量是深度的，因为它们是biLM所有内部层的函数**。更具体地说，我们学习了堆积在每个终端任务的每个输入词向量上的的线性组合，这比仅使用顶层LSTM层的性能更好。

 &emsp;&emsp; 以这种方式组合内部的state可以实现非常丰富的单词表示。使用内在的评价，我们发现，**高阶的LSTM state捕获词义的上下文相关的信息(例如，他们可以不需要修改任何东西，就可以在监督词义消歧任务上表现良好)，低阶的state 捕获语法相关的信息(例如，他们可以用来做词性标注)。**同时捕获所有这些信号是非常有益的，对于每一个终端任务来说，这允许可学习模型选择最有用的半监督模型。

 &emsp;&emsp; 大量的实验表明，ELMo表示法在实践中非常有效。我们首先表明，对于六种不同且具有挑战性的语言理解问题，它们可以很容易地添加到现有的模型中，包括文本隐含、问题回答和情绪分析。在任何情况下，仅添加ELMo表示就显著提高了技术水平，包括高达20%的相对错误率下降。对于可能进行直接比较的任务，ELMo的表现优于CoVe  (McCann et al.， 2017)， CoVe使用神经机器翻译编码器计算上下文化表示。最后，对ELMo和CoV e的分析表明，deep  representation优于仅来自LSTM顶层的那些。我们的训练模型和代码是公开的，我们期望ELMo将为许多其他的NLP问题提供类似的收益

## 2、 Related work

 &emsp;&emsp; 由于预训练的词向量能够从大规模的未标记文本中捕获单词的句法和语义信息，因此他们(Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014)是大多数最新的NLP架构的标准组件，包括回答问题(Liu et al.，  2017)、文本蕴涵(Chen et al.， 2017)和语义角色标注(He et al.，  2017)。但是，这些学习单词向量的方法只允许对每个单词使用独立于上下文的representation。

 &emsp;&emsp; 之前提出的方法通过添加子词（subword）信息来丰富传统词向量(例如，Wieting et al., 2016; Bojanowski et al., 2017)或学习每个词意义的单独向量(例如，Neelakantan et al., 2014)，克服了传统词向量的一些缺点。**通过字符卷积（ character convolutions）的使用，我们的方法也受益于子词单元，并且我们无缝地将多个任务的信息融入到一个下游任务中，而不需要明确的对该下游任务进行单独的预训练。**

 &emsp;&emsp; 其他近期的工作也集中在学习情境依赖的表征上。context2vec (Melamud et al.，  2016)使用了双向长短期记忆(LSTM;Hochreiter和Schmidhuber,  1997)围绕一个中心词编码上下文。其他学习上下文词向量的方法包括在表示中的枢轴词本身，并通过监督神经机器翻译(MT)系统(CoV e; McCann et al., 2017或无监督语言模型(Peters et al., 2017)进行计算。这两种方法都受益于大数据集，尽管MT方法受到平行语料库规模的限制。在本文中，我们充分利用对丰富的单语数据的访问，并在大约3000万句的语料库上训练我们的biLM  (Chelba et al.， 2014)。我们还将这些方法推广到深度情境表征，我们表明这些方法在广泛的各种NLP任务中都能很好地发挥作用。

 &emsp;&emsp; 之前的研究也表明，不同层的biRNNs可以编码不同类型的信息。例如，引入多任务句法监督(如词性标记)的低阶LSTM可以提高更高级别的任务的总体性能，比如依赖性解析(Hashimoto et al., 2017)或CCG super tagging (Søgaard and Goldberg,2016)。在基于rnnn的编解码机翻译系统中，Belinkov等人(2017)表明，在2层LSTM编码器中，第一层学习的表示比第二层更能预测POS标签。最后，用于编码单词上下文的LSTM  (Melamud et al.，  2016)的顶层被证明用于学习单词意义的表示。我们发现，我们可以通过修改我们的ELMo表征的语言模型目标来引入这些类似的signal，混合不同类型的半监督任务，对于为下游任务建模来说是非常有益的。

 &emsp;&emsp; Dai and Le(2015)和Ramachandran et  al.(2017)使用语言模型和序列自编码器对encoder-decoder进行预训练，然后根据具体任务进行微调。相反，在使用未标记数据对biLM进行预训练后，我们确定了权重，并添加了额外的特定任务模型容量，从而允许我们在下游训练数据规模指示较小的监督模型的情况下，利用大型、丰富和通用的biLM表示。

## 3、ELMo: Embeddings from Language Models

 &emsp;&emsp; 与大多数最常用的词向量不同，ELMo词向量是整个输入句子的函数，在这一节会进行讲解。他们会在最初的两层biLMs中进行训练(第3.1节)，biLMs是使用character CNN构建的，是内部网络state的线性函数(第3.2节)。这个步骤让我们可以做半监督学习，其中biLM是我们在大规模的数据集上提前训练好的(第3.4节)，并且很容易地被整合到大量现有的神经NLP架构中(第3.3节)。

### 3.1 Bidirectional language models

 &emsp;&emsp; 给定带有$\ N$ 个token的序列，$\ (t_1, t_2, ..., t_N)$ ，给定$\ (t_1,...,t_{k-1})$ ，前馈语言网络计算$\ t_k$ 的概率。即：
$$
p(t_1,t_2,...,t_N) = \prod_{k=1}^N p(t_k|t_1,t_2,...,t_{k-1})
$$
 &emsp;&emsp; 目前，SOTA的神经语言模型(Józefowicz et al., 2016; Melis et al., 2017; Merity et al., 2017)计算与上下文无关的token词向量$\ x_k^{LM}$ ，然后将其通过L层的正向LSTM。在每个位置$\ k$ ，每个LSTM层都会输出一个依赖于上下文的词向量$\ \vec{h}_{k,L}^{LM}$ ，被用来去预测下一个token$\ t_{k+1}$ （使用Softmax）。

 &emsp;&emsp; 反向LM与正向LM是很像的，只不过反向LM是在反向的序列中运行，即，给定未来的上下文，来预测当前的词：
$$
p(t_1,t_2,...,t_N) = \prod_{k=1}^N p(t_k | t_{k+1},t_{k+2},...,t_N)
$$
 &emsp;&emsp; 反向LM可以用类似前向LM的方式实现，在反向LM中，每一个反向LSTM的第$\ j$ 层就是根据$\ (t_{k+1},...,t_N)$ 来生成representation$\ \overleftarrow{h}^{LM}_{k,j}$  。

 &emsp;&emsp; biLM结合了前向LM和反向LM，我们的公式就是最大化前向和反向的log似然，即：
$$
\sum_{k=1}^N(\log p(t_k|t_1,...,t_{k-1};\Theta_x,\vec{\Theta}_{LSTM},\Theta_s) + \log p(t_k|t_{k+1},...,t_{N};\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s))
$$
 &emsp;&emsp; 其中，$\ \Theta_x$ 表示词向量，$\ \Theta_s$ 表示Softmax层，在前向LM和反向LM中，我们使这两个参数相同，但是其他的参数是不同的，保持独立。总的来说，这个方法与 Peters et al. (2017)的方法很像，但是，我们在不同的方向上共享了权重。在下一节中，我们将介绍一个新的方法来学习词向量，这是一种biLM层的线性组合。

### 3.2 ELMo

 &emsp;&emsp; ELMo是biLM中中间层representation的任务特定组合，对于每一个token，一个$\ L$ 层的biLM能够得到$\ 2L+1$ 个representation，其集合如下：
$$
\begin{equation}
\begin{split}
R_k &= \{x_k^{LM},\vec{h_{k,j}^{LM}}, \overleftarrow{h_{k,j}^{LM}} | j = 1,...,L\}\\
&={h_{k,j}^{LM} | j = 0,...,L}
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中，$\ h_{k,0}^{LM}$ 是token层的词向量，$\ h_{k,j}^{LM} = [\vec{h_{k,j}^{LM}}, \overleftarrow{h_{k,j}^{LM}]}$ 表示每一个biLSTM层的输出。

 &emsp;&emsp; 为了方便将我们的模型融入到下游模型中，ELMo将R中的所有向量，整合成一个单独的向量，即$\ ELMo_k = E(R_k;\Theta_e)$ 。在最简单的情况下，ELMo会只选择前面几层。在TagLM(Peters et al., 2017)和CoVe(McCann et al., 2017)中，$\ E(R_k) = h_{k,L}^{LM}$ 。更一般的，我们计算biLM中所有的层的权重，即：
$$
ELMo_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k,j}^{LM} \tag{1}
$$
 &emsp;&emsp; 在$\ (1)$ 中，$\ s^{task}$ 使softmax-normalized权重（总和为1），标量参数$\ \gamma^{task}$ 允许任务模型缩放整个ELMo向量。$\ \gamma$ 是非常重要的，可以辅助优化过程。考虑到每个biLM层的激活函数都有不同的分布，在某些情况下，在每个biLM层前做layer normalization通常是有帮助的。

### 3.3 Using biLMs for supervised NLP tasks

 &emsp;&emsp; **对于一个有目标的NLP任务，给定一个预训练的biLM和一个监督任务的架构，使用biLM来改进任务模型是一个很简单的操作。我们只需运行biLM并记录每个单词的所有层representation。然后，我们让终端任务模型学习这些表示的线性组合**，如下所述：

 &emsp;&emsp; 首先考虑不使用biLM的监督模型的最低层。大多数监督的NLP模型在最低层共享一个公共架构，允许我们以一致的、统一的方式添加ELMo。给定一个token序列$\ (t_1，…,t_N)$ ，在每个token位置上，我们通常会得到一个与上下文无关的词向量$\ x_k$  ，之后，模型会得到一个上下文敏感的representation$\ h_k$ ，通常，我们使用双向RNN、CNN或前馈神经网络来得到。

 &emsp;&emsp; 为了把ELMo加入到监督模型中，我们会首先冻结biLM的权重，然后将ELMo向量$\ ELMo_k^{task}$ 与词向量$\ x_k$ concat在一起，将得到的增强到representation$\ [x_k;ELMo_k^{task}]$ 放到RNN任务中去。对于某些任务来说（例如，SNLI，SQuAD），我们发现，把ELMo加入到RNN任务的输出中，也能够提升性能，即，最后得到的向量为$\ [h_k;ELMo_k^{task}]$ 。监督模型剩下的部分保持不变。例如，看一下在第4节中的实验，在biLSTM后跟着一个bi-attention层，或者在coreference resolution实验中，在biLSTM后会跟着聚类模型。

 &emsp;&emsp; 最后，我们发现在EMLo后面加适当的dropout是有益的，在某些情况下，**为了正则化ELMo权重，我们可以将$\ \lambda||w||_2^2$ 加入到loss中**。这给ELMo的权重添加了一些偏差，使其更加接近所有biLM层的平均值。

### 3.4 Pre-trained bidirectional language model architecture

 &emsp;&emsp; 本文预训练的biLMs与Jozefowicz等人(2016)和Kim等人(2015)的架构相似，但进行了修改，支持正向和反向的联合训练，并**在LSTM层之间增加了一个残差连接**。在这项工作中，我们关注的是大规模的biLMs，正如Peters等人(2017)强调的，使用biLMs比只有前向LMs和大规模数据集训练更重要。

 &emsp;&emsp; 为了保持模型的输入只有基于字符的词向量，并平衡整体语言模型的复杂性和模型大小以及下游任务的计算需求，我们将单一最佳模型 CNN-BIG-LSTM （Józefowicz et al. (2016). ）中的嵌入层和隐藏层维数减半。最终的模型使用$\ L =  2$ 个biLSTM层，有4096个单元和512维投影，以及从第一层到第二层的残差连接。上下文不敏感的representation使用2048个 n-gram 字符卷积过滤器，然后是两个 highway 层(Srivastava  et al.，  2015)和一个线性投影层（投影到到512维）。因此，biLM为每个输入token提供了三层表示，包括那些在训练集外的纯字符输入。相比之下，传统的单词嵌入方法在固定词汇表中只提供token的一层表示。

 &emsp;&emsp; 在1B的基准测试中训练10个epoch，得到的前向和反向的困惑度（perplexities）为39.7，前向CNN-BIG-LSTM的perplexities为30。一般情况下，我们发现正向perplexities和逆向perplexities近似相等，而逆向perplexities略低。

 &emsp;&emsp; 在预训练完成后，biLM可以为任何一个任务计算representation。在某些情况下，对领域特定数据进行biLM微调可以显著降低复杂性，并提高下游任务的性能。这可以看作是biLM的一种领域迁移。因此，在大多数情况下，我们在下游任务中使用一个经过调优的biLM。详见补充材料。

## 4、Evaluation

 &emsp;&emsp; 表1显示了ELMo在6个不同的基准NLP任务集上的性能。在考虑的每一个任务中，简单地添加ELMo就可以建立一个新的最先进的结果，相对于强基础模型，误差减少6 -  20%。对于不同的set模型架构和语言理解任务来说，这是一个非常普遍的结果。在本节的其余部分中，我们将提供单个任务结果的高级简述（sketch）;完整的实验细节见补充材料。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/2.png?raw=true)

### 4.1 Question answering

 &emsp;&emsp; The Stanford Question Answering Dataset (SQuAD) (Rajpurkar et al., 2016) 包含了100K+的question - answer 对，其中答案是维基百科中给定段落中的一部分。我们的baseline(Clark and Gardner,  2017)是对Seo等人的Bidirectional Attention Flow model(BiDAF;2017)的提升。我们在bidirectional attention之后增加了一个self-attention层，简化了一些池操作，并将LSTMs替换为门控周期性单元(GRUs;Cho等，2014)。在baseline中加入ELMo后，测试集$\ F_1$ 分数从81.1%提高到85.8%，相对误差降低了24.9%，整体提高了1.4%。由11名成员组成的团队将$\ F_1$ 推至87.4，这是提交到排行榜时的整体水平。使用ELMo时4.7%的增长也明显大于在baseline中加入CoVe后1.8%的增长(McCann  et al.， 2017)。

 &emsp;&emsp; 其他的各项任务与对应的数据集如下：

- Textual entailment -- SNLI
- Semantic role labeling --  SRL
- Coreference resolution -- Coref
- Named entity extraction  --  NER
- Sentiment analysis  -- SST-5



## 5、Analysis

 &emsp;&emsp; 本节提供消融分析，以验证我们的主要声明，并阐明ELMo representation的一些有趣的方面。第5.1节表明，在下游任务中使用深度上下文表示比之前仅使用顶层的工作提高了性能，无论它们是由biLM还是MT编码器生成的，而且ELMo representation提供了最佳的总体性能。第5.3节探究了在biLMs中捕获的不同类型的上下文信息，并使用两种内在评估来表明，语法信息在较低的层次上得到更好的表示，而语义信息在较高的层次上得到表示，这与MT编码器一致。它还表明，我们的biLM始终提供比CoV  e更丰富的表示法。此外，我们分析了任务模型中包含ELMo的位置(5.2节)、训练集大小(5.4节)的敏感性，并可视化了各个任务的ELMo学习权重(5.5节)。

### 5.1 Alternate layer weighting schemes

 &emsp;&emsp; **我们有很多可供选择的公式1方案，来连接biLM的各个层**。之前的关于上下文representation的工作只使用了最后一层，无论是biLM(Peters et al., 2017)还是MT编码器（CoV e; McCann et al., 2017）。**正则化参数$\ \lambda$ 的选择也是很重要的，当$\ \lambda = 1$ 时，会降低权重函数的复杂性，甚至是取简单的平均，在正则化参数比较小时，例如$\ \lambda = 0.001$ ，这就使得层的权重更加多样性。**

  &emsp;&emsp; 表2在SQuAD、SNLI和SRL数据集上比较了这些可选方案。**相比于只使用最后一层来说，使用所有层的表示可以提高整体性能，而使用最后一层的上下文表示也可以提高basline的性能。**例如，在SQuAD的案例中，仅仅使用最后一层biLM就可以使$\ F_1$ 比baseline提高3.9%。与“仅使用最后一层”相比，平均所有的biLM层（$\ \lambda = 1$ ），而不是只使用最后一层，将$\ F_1$ 提高0.3%，“$\ \lambda = 0.001$ ”将使任务模型学习各个层的权重，将$\ F_1$ 提高0.2%($\ \lambda = 1$   vs.$\ \lambda = 0.001$)。在大多数情况下，ELMo更偏爱比较小的$\ \lambda$ 。数据集比较小的情况下，比如NER， 结果对$\ \lambda$ 更加敏感。

 &emsp;&emsp; 总体趋势与CoVe相似，但在baseline上的增幅较小。对于SNLI来说，与只使用最后一层相比，使用全平均化的方法（$\ \lambda = 1$ ）将开发精度从88.2%提高到88.7%，相比于只使用最后一层的情况，SRL的 $\ F_1$ （$\ \lambda = 1$ ）为增加了0.1%，达到82.2，相比于只是用最后一层的情况。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/3.png?raw=true)

### 5.2 Where to include ELMo?

 &emsp;&emsp; 本文中所有的任务架构都只将词向量作为biRNN最低层的输入。然而，**我们发现在特定任务架构中，在biRNN的输出中包含ELMo可以改善某些任务的总体效果。**如表3所示，在SNLI和SQuAD数据集上，**在输入和输出层都包括ELMo**，比仅输入层加入ELMo有所提升，但是对于SRL(和coreference resolution，没有显示在表三中)，当它只包含在输入层时，性能是最高的。一个可能的解释是SNLI和SQuAD架构都在biRNN之后使用注意力层，所以在这一层引入ELMo使得模型可以直接关注biLM的内部表示。而在SRL中，特定于任务的上下文表示可能比来自biLM的上下文表示更重要。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/4.png?raw=true)

### 5.3 What information is captured by the biLM’s representations?

 &emsp;&emsp; 相比于单独的词向量来说，添加ELMo模块提升了任务的性能，也就是说，biLM的语境representation一定编码了词向量中没有捕获的更有用的信息。直观的说，**biLM必须通过上下文来消除词语含义的歧义**。考虑“play”，一个高度多义词。表4的顶部列出了使用GloVe向量“play”的最近邻居。它们分布在不同的词性中(例如，“played”，用作动词的“playing”，“player”，作为名词的“game”)，但集中在与体育运动相关的“play”。相反，下面两行显示了来自SemCor数据集(见下面)的最邻近的句子，使用源句子中biLM的上下文表示“play”。在这些情况下，biLM能够消除源句中词性和词义的歧义。

 &emsp;&emsp; 类似于Belinkov等人(2017)，这些观察可以通过对上下文表征的内在评估来量化。为了分离由biLM编码的信息，使用representation直接对细粒度的单词词义消歧(WSD)任务和词性标记任务进行预测。使用这种方法，还可以比较CoVe，并跨越每个单独的层。

**Word sense disambiguation**

 &emsp;&emsp; 给定一个句子，我们可以使用biLM使用简单的最近邻法预测目标词的意义，类似于Melamud等人(2016)。为此，我们首先使用biLM来计算SemCor  3.0(我们的训练语料库)中所有单词的表示(Miller et al.，  1994)，然后取每种意义的平均表示。在测试时，我们再次使用biLM来计算给定目标词的表示，并从训练集取最近邻意义，对于训练期间未观察到的词元，从WordNet取第一意义。

 &emsp;&emsp; 表5使用Raganato等人(2017b)的评估框架比较了WSD的结果，该评估框架使用了Raganato等人(2017a)的同一套四个测试集。总的来说，biLM顶层的表面层的F1值为69.0，在WSD上比第一层表现得更好。这与使用手工特性的最先进的特定于wsd的监督模型(Iacobacci et al6, 2016)和使用辅助粗粒度语义标签和POS标记训练的特定于任务的biLSTM (Raganato et al, 2017a)形成了竞争。CoVe biLSTM层的模式与来自biLM层的模式类似(第二层的总体性能比第一层更高)，但是，我们的biLM优于CoVe biLSTM，后者落后于WordNet first sense baseline.

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/5.png?raw=true)

**POS tagging**

 &emsp;&emsp; 为了检验biLM是否捕获了基本语法，我们使用上下文表示作为线性分类器的输入，该分类器使用Penn Treebank的华尔街日报部分预测POS标记(PTB Marcus et al, 1993)。由于线性分类器只增加了少量的模型容量，这是对biLM表示的直接检验。与WSD类似，biLM表示与经过仔细调优的特定任务的biLSTMs具有竞争性(Ling et al, 2015, Ma和Hovy, 2016)。但是,与WSD,精度使用第一个biLM层高于顶层,深biLSTMs一致的结果在多任务训练(Goldberg等Søgaard和Hashimoto ,2016年,2017年)和(Belinkov等,2017)。CoVe POS标签精度遵循与来自biLM相同的模式，就像WSD一样，biLM实现了比CoVe编码器更高的精度。

**Implications for supervised tasks**

 &emsp;&emsp; 综上所述，这些实验证实了biLM中的不同层代表不同类型的信息，并解释了为什么包含所有的biLM层对于下游任务的最高性能非常重要。此外，与CoVe相比，biLM的表示更适合于WSD和POS标记，这有助于说明为什么ELMo在下游任务中表现优于CoVe。

### 5.4 Sample efficiency

 &emsp;&emsp; 将ELMo添加到模型中可以极大地提高样本效率，无论是为了达到最新性能而进行的参数更新的数量，还是整个训练集的大小。添加ELMo之前，我们需要训练486个epoch才能够达到F1分数的最大值，添加ELMo后，模型在10个epoch时超过了baseline的最大值，达到相同性能水平所需的更新数量相对减少了98%。

 &emsp;&emsp; 此外，与没有ELMo的模型相比，ELMo增强的模型更有效地使用更小的训练集。图1比较了有和没有ELMo的基线模型的性能，因为整个训练集的百分比从0.1%到100%不等。ELMo最大的改进是针对更小的训练集，并显著减少了达到给定性能水平所需的训练数据量。在SRL的情况下，包含1%训练集的ELMo模型与包含10%训练集的baseline模型的F1值大致相同。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/6.png?raw=true)

### 5.5 Visualization of learned weights

 &emsp;&emsp; 图2显示了softmax规范化的学习层权重。在输入层，对于coreference和SQuAD来说，任务模型倾向于第一个biLSTM层。但是对于其他任务来说，它的分布并没有达到峰值。输出层的权重相对平衡，较低的层略有偏好。

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E5%9B%9B%EF%BC%89ELMo/7.png?raw=true)

## 6、Conclusion

 &emsp;&emsp; 我们介绍了一种从biLMs中学习高质量的上下文相关表示的通用方法，并在将ELMo应用于广泛的NLP任务时展示了巨大的改进。通过ablations和其他控制实验，我们也证实了biLM层可以有效地编码关于words-in-context的不同类型的语法和语义信息，并且使用所有层可以提高整体任务性能。