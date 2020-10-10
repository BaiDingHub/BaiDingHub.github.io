---
title: NLP特征提取器篇（三）Transformer
date: 2020-09-12 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP特征提取器篇（三）Transformer"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# Attention Is All You Need
> 时间：2017年
>
> 关键词：NLP，Transformer
>
> 论文位置：https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
>
> 引用：Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.

**摘要：**目前，序列转换模型主要是基于复杂的RNN或CNN模型，模型中包括一个编码器和一个解码器。目前，性能最好的模型是利用注意力机制来连接编码器和解码器。我们**提出了一个简单的网络结构，Transformer，它完全基于注意力机制**，摒弃了RNN和CNN。在两个机器翻译任务上的实验表明，这些模型**在性能上有优势，同时具有更强的并行性，并且需要的训练时间显著减少**。我们的模型在WMT 2014 English-to-German translation 任务中达到28.4 BLEU，比现有的最佳结果(包括集成)提高了2BLEU以上。在WMT 2014 English-to-French translation task任务中，我们的模型在8个GPU上经过3.5天的训练后，建立了一个新的SOTA的单模型，BLEU得分41.0，这只是文献中最好模型的训练成本的一小部分。

**索引**- 自然语言处理，

## 内容分析

  &emsp;&emsp; RNN本身的序列依赖结构对于大规模并行计算来说相当不友好，$\ S_t$ 的计算需要$\ S_{t-1}$ 已经计算好才行，这也就导致RNN的并行计算能力比较差。Transformer沿用了encoder-decoder架构，在Transformer的架构中，摒弃了RNN，采用完全基于attention的方法来构建，解决了RNN的并行能力问题，同时，attention的使用，提高了模型的能力。

 &emsp;&emsp; Transformer本质上是一个encoder-dencoder结构，在结构中大量的使用了attention，其主要结构有Multi-Head Attention、Skip connection、前向神经网络、Positional Encoding，其具体结构如下，每个结构的解析已经在第三节中讲解：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/1.png?raw=true)



### 1）Transformer的训练

 &emsp;&emsp; 以机器翻译任务为例，讲解Transformer的训练，给定数据`I am fine` ，对应的预测值`我很好`。

**数据输入处理**

 &emsp;&emsp; 对输入数据分词，得到对应的token，转换成one-hot，然后根据词向量矩阵，我们就得到了输入数据的词向量，假设词向量维度为512，同时，令我们的序列最大长度为10，那么，我们得到的数据就是$\ (10,512)$ 。剩下空白的部分用padding的embedding填充，注意padding是一个特有的东西，用来表示空白，其有其自己的mebdding。

 &emsp;&emsp; 对于词`I` ，其位置信息`pos = 0` ，我们将该pos利用正旋编码解析成embedding，维度为512，有三个词，我们就得到了三个这样的embedding。

 &emsp;&emsp; 然后把得到的位置的embedding即Positional Encoding，加到对应词的词向量中，这样就得到了我们的输入数据，维度为$\ (10,512)$ 。

**Encoder处理数据**

 &emsp;&emsp; 如果，我们把一个Multi-head Attention以及一个前馈神经网络当成一个单元的话，Encoder总共有$\ N$ 个单元堆叠起来，这里，我们取$\ N = 6$ 。

 &emsp;&emsp; 我们的输入数据$\ (10,512)$ 首先输入Multi-head Attention，在Multi-head Attention中，首先有三个线性层，将我们的输入转换成了Attention所必要的Q、K、V向量，维度分别为$\ (10,d_k),(10,d_k),(10,d_v)$ ，也就是说，Q、K的维度是一样的。之后呢，将Q、K、V送入Scaled Dot-Product Attention，得到输出，输出为$\ (10,d_v)$ 维的矩阵。

 &emsp;&emsp; 由于作者采用了Multi-head方法，因此，我们会有$\ h$ 个上面的attention结构，每个attention的输入都是$\ (10,512)$ ，得到的输出都是$\ (10, d_v)$ 。

 &emsp;&emsp; 之后，将这$\ h$ 个输出concat在一起，得到的矩阵维度为$\ (10,hd_v)$ 。在实验中，选择$\ h = 8$ ，$\ d_k = d_v = 512/h = 64$ 。因此，经过Multi-head Attention后得到的输出为$\ (10,512)$ 。这也是为了方便之后的残差连接。在得到输出后，我们可以对输出进行dropout，当然也可以没有。

**Add & Norm**

 &emsp;&emsp; 我们得到了$\ (10,512)$ 的输出，之后就可以与输入进行残差连接，然后进行layer normliaztion。

**前馈神经网络处理**

 &emsp;&emsp; 这一步的输入依然是$\ (10,512)$ ，我们在这一层添加了两个线性层和一个ReLU层，其表达式为：
$$
FFN(x) = \text{max}(0,xW_1+b_1)W_2 + b_2
$$
 &emsp;&emsp; 经过这一层后，输出为$\ (10,512)$ 。之后经过同样的Add & Norm。得到这一个单元的输出$\ (10,512)$ 。将这个单元的输出作为下一个单元的输入，这样重复$\ N = 6$ 次。

**Decoder输入**

 &emsp;&emsp; 我们可以看到，在右侧部分也有一个输入，这个输入代表了预测值的信息。Transformer是怎么进行预测的呢？我们根据输入`I am fine` ，先预测翻译值`我` ，之后预测`很`，然后预测`好` ，所以，我们的运行过程是运行$\ N$ 次Transformer（N就是预测值的最大序列长度，剩下的部分预测为padding），来逐步的得到我们的预测值。

 &emsp;&emsp; 因此，训练阶段，我们可以利用当前需要预测的值的位置信息以及之前的数据信息来进行训练，这个`Output Embedding` 就是`我,很，好,...,Padding`，那么，我们怎么保证，当前的预测只使用当前位置之前的信息呢？采用mask操作，即将当前位置以及当前位置之后的信息都mask掉，mask其实就是一个0，1矩阵，在第一次预测时为`[0,0,...,0]` ，第二次为`[1,0,...,0]` ，第三次为`[1,1,...,0]` 等等，这个mask操作是在Decoder的第一个子层中才用到的，这里提前说了。

**Decoder第一个子层**

 &emsp;&emsp; Decoder的这个子层其实跟Encoder的第一个子层一致，只是加了mask操作。所以输入了预测值的信息，即$\ (10,512)$ 的矩阵后，输出也是$\ (10,512)$ 。

**Decoder第二个子层**

 &emsp;&emsp; 这个子层就不一样了，在这一层中，将上一层的输入作为$\ V$ 值。然后，将与该Decoder对应位置的Encoder的输出作为$\ Q,K$ 值，然后进行同样的操作，输出为$\ (10,512)$ 。

**Decoder的剩余部分**

 &emsp;&emsp; 之后，将输出送入前馈神经网络，重复这样$\ N$ 次。得到输出$\ (10,512)$ 。

**最终输出**

 &emsp;&emsp; 将$\ (10,512)$ 的输出经过一个线性层，然后经过Softmax，就得到了我们的概率输出，就能够预测当前位置的预测值了。

### 2）Transformer的预测

 &emsp;&emsp; 预测阶段与训练阶段几乎一致，只有一处不同。因为测试时，我们是没有预测值的，因此Decoder的输入应该为`Padding,...,Padding`。

### 3）Transformer的优势

- 突破了 RNN 模型不能并行计算的限制。
- 相比 CNN，计算两个位置之间的关联所需的操作次数不随距离增长。
- 自注意力可以产生更具可解释性的模型。我们可以从模型中检查注意力分布。各个注意头(attention head)可以学会执行不同的任务。

## 1、Introduction

 &emsp;&emsp; RNN，特别是LSTM[12]和GRU[7]，在序列建模和转换问题(如语言建模和机器翻译)中已经被牢牢地确立为SOTA的方法[29,2,5]。此后，大量的努力不断地推动循环语言模型和编码器-解码器架构的边界[31,21,13]。

 &emsp;&emsp; **递归模型**通常沿着输入和输出序列的符号位置进行因子计算。其位置是与时间对齐的，他们生成一个隐藏的状态序列$\ h_t$ ，$\ h_t$ 是前面的函数隐藏状态$\ h_{t−1}$ 和位置$\  t$ 的输入$\ x_t$ 的函数。**这种固有的顺序特性排除了训练示例中的并行化的可能，而在较长的序列长度下，并行化就变得至关重要，因为内存约束限制了示例之间的批处理。**最近的工作通过因子分解技巧[18]和条件计算[26]在计算效率上取得了显著的提高，同时也提高了后者的模型性能。然而，**顺序计算的基本约束仍然存在**。

 &emsp;&emsp; 注意力机制已经成为各种任务中引人注目的序列建模和转换模型的一个不可分割的部分，**其允许建模依赖关系而不需要考虑输入或输出序列的距离**[2,16]。然而，除了少数情况下，几乎所有的注意力机制都是与RNN一起使用的[22]。

 &emsp;&emsp; 在这篇文章中，我们提出了Transformer，一种摒弃了RNN的模型架构，它完全依赖于注意力机制来建立输入和输出的全局依赖关系。同时，Transformer实现了极大程度的并行化，在翻译任务中的性能达到了SOTA，同时在八个P100上只需要训练12个小时。

## 2、Background

 &emsp;&emsp; 减少序列运算时间的这个目标催生了Extended Neural GPU[20], ByteNet [15] and ConvS2S [8]，这些模型都是使用CNN作为基本模块，并行计算所有输入和输出位置的隐藏representation。在这些模型中，**将来自两个任意输入或输出位置的信号联系起来所需的操作数量随着位置之间的距离增加，ConvS2S为线性增加，ByteNet为对数增加**。这使得了解距离较远的位置[11]之间的依赖关系变得更加困难。**在Transformer中，这被减少为一个恒定的操作次数**，尽管其代价是由于平均注意力加权位置而降低了有效分辨率，正如3.2节中所述，我们**使用mulit-head注意力来抵消这种影响**。

 &emsp;&emsp; **Self-attention有时候也被称为 intra-attention ，是一种将单个序列中的不同位置联系起来的注意力机制，并能够计算该序列的representation**。Self-attention已被成功地应用于各种任务，包括阅读理解、摘要总结、文本蕴涵和学习独立于任务的句子表征。

 &emsp;&emsp; **基于循环注意力机制的端到端记忆网络**已被证明在简单语言问题回答和语言建模任务[28]上表现良好。

 &emsp;&emsp; 据我们所知，Transformer是第一个完全依赖Self-attention来计算输入和输出representation而不使用序列对齐的RNNs或卷积的转换模型。在接下来的章节中，我们将描述Transformer，motivate self-attention，并讨论其优于诸如[14,15]和[8]等模型的优点。

## 3、Model Architecture

 &emsp;&emsp; 大多数性能比较强的序列转换模型都有encoder-decoder结构[5,2,29]，在模型中，encoder将输入序列$\ (x_1,...,x_n)$ 映射到$\ z = (z_1,...,z_n)$ 。给定$\ z$ ，decoder生成一个输出序列$\ (y_1,...,y_m)$ 。每一步，模型都是自回归的，即在生成下一个符号之前，都会用到上一步中生成的符号（就是指的RNN中的循环）。

 &emsp;&emsp; Transformer遵循了这样的架构，为encoder和decoder使用了堆叠的Self-attention和point-wise的全连接层，分别在图一的左半部分和右半部分展示：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/1.png?raw=true)

### 3.1  Encoder and Decoder Stacks

#### 3.1.1 Encoder

 &emsp;&emsp; encoder是由$\ N = 6$ 个相同的层组成，**每一层包括两个子层，第一个是multi-head self-attention mechanism，第二个是一个简单的position-wise的全连接神经网络**。在每一层后面我们都采用了**残差连接**，同时使用了**layer normalization**。因此，每一个子层的输出都可以表示为$\ \text{LayerNorm}(x+\text{Sublayer}(x))$ ，其中$\ \text{SubLayer}(x)$ 表示子层的函数。同时，为了进行残差连接，**模型中的所有子层包括embedding层，他们的输出维度都是$\ d_{\text{model}} = 512$ 。**

#### 3.1.2 Decoder

 &emsp;&emsp; decoder也是由$\ N = 6$ 个相同的层组成。除了encoder层中有的两个子层，decoder层加入了第三个子层，其在对应层的encoder的输出上执行multi-head attention。正如encoder一样，我们在每一个子层后都进行了残差处理和layer normalization。我们还修改了decoder中的self-attention子层。利用mask，使得当前位置不会注意到后面的位置信息。mask操作确保了位置$\ i$ 上的预测仅仅依赖于$\ i$ 前的已知的输出。同时，在训练阶段，decoder的输入是$\ y$ 所对应的token，如果是测试阶段，该处输入为padding。

### 3.2 Attention

 &emsp;&emsp; 注意力机制可以描述为将query和一组key-value对映射到输出，其中query、key、value和输出都是向量。输出是values的加权和，其中分配给每个value的权重由query与相应key的compatibility函数计算。

#### 3.2.1 Scaled Dot-Product Attention

 &emsp;&emsp; 我们把我们独特的注意力机制称之为“**Scaled Dot-Product Attention**”，其**输入包括维度为$\ d_k$ 的queries，维度为$\ d_k$ 的key，维度为$\ d_v$ 的value。我们计算query与所有的key的点积，将计算的结果除以$\ \sqrt{d_k}$ ，之后使用softmax函数来获得value的权重**。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/2.png?raw=true)

 &emsp;&emsp; 实际上，我们会在一组query上同时计算attention函数，将这组query整合成矩阵Q，key和value也被打包成矩阵$\ K$ 和$\ V$ ，那么，我们的矩阵的输出为：
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
 &emsp;&emsp; 两种最常用的attention函数使additive attention[2]以及dot-product attention。**Dot-product attention跟我们的算法是一样的，除了没有归一化因子$\ \frac{1}{d_k}$** ，additive attention使用带有一个隐藏层的前馈神经网络来计算compatibility函数。尽管，这两种方法在理论上复杂度相似，但是实际上dot-product会更快一些，也更加省空间，因为，我们可以使用高度优化的矩阵乘法代码来实现它。

 &emsp;&emsp; 当$\ d_k$ 比较小时，两种方法性能差不多，但是$\ d_k$ 比较大时，Dot-product就不如additive attention表现的好，我们怀疑对于大的$\ d_k$ 来说，点积的值会变大，从而将softmax函数推入其梯度极小的区域。为了抵消这个影响，我们将点积乘以$\ \frac{1}{\sqrt{dk}}$ 。

#### 3.2.2 Multi-Head Attention

 &emsp;&emsp; **相比于仅进行单个attention函数（query、key、value维度都是$\ d_{model}$ ）来说，我们发现，使用不同的、可学习的线性投影，将query、key、value投影h次到$\ d_k,d_k,d_v$ 维度相对来说更好一些。我们在每一个投影得到的query、key、value上并行的执行attention，每一个attention函数都得到一个$\ d_v$ 的输出值，最后我们把这些输出值concat起来，再经过一次投影**，就得到了我们最后的结果，如图2所示。

 &emsp;&emsp; **multi-head attention允许模型关注不同的位置，从而在不同的子空间中学习到不同的representation**，而如果我们只有一个attention的话，我们就只能得到这些representation的平均。multi-head的数学表示为：
$$
\begin{equation}
\begin{split}
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,...,\text{head}_h) W^O\\
\text{where}\ \text{head}_i&= \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中，参数矩阵$\ W_i^Q \in \mathbb{R^{d_{\text{model}} \times d_k}}, W_i^K \in \mathbb{R^{d_{\text{model}} \times d_k}}, W_i^V \in \mathbb{R^{d_{\text{model}} \times d_v}}, W_i^O \in \mathbb{R^{hd_v \times d_{\text{model}}}} $ 表示投影层。

 &emsp;&emsp; 在我们的工作中，我们采用$\ h = 8$ ，同时，我们采用$\ d_k = d_v = d_{\text{model}}/h = 64$ 。由于，我们每个head的输出维度$\ d_v$ 都减少了，所以总的计算代价跟单个attention几乎相同。

#### 3.2.3 Applications of Attention in our Model

 &emsp;&emsp; Transformer使用了三种不同的方式使用multi-head attention：

- 在encoder-decoder attention层（decoder层的中间一层）中，query来自于上一个decoder层，key和value来自于encoder层的输出，这就使得我们可以在decoder的每个部分都具备输入序列的所有位置信息，这模仿了传统seq2seq模型的encoder-decoder attention机制，例如[31,2,8]
- encoder中包含self-attention层，在self-attention层中，所有的value、key、suery都来自同一个地方，即encoder中上一层的输出。
- 同样的，在decoder层中的self-attention层使得decoder中的每个部分都具备输入序列的所有位置信息。同时，为了保持decoder中的自回归特性，我们需要阻止左侧的信息流入decoder。我们通过对attention中softmax的输入部分进行mask，即将非法连接的值设置为负无穷，解决了这个问题，见图二。

### 3.3 Position-wise Feed-Forward Networks

 &emsp;&emsp; 除了attention层外，我们还包含了全连接层，它分别应用于每一个位置，采用相同的操作，包括两层线性变换，以及一个ReLU函数：
$$
FFN(x) = \text{max}(0,xW_1+b_1)W_2 + b_2
$$
 &emsp;&emsp; 尽管在不同的位置上采用了相同的线性变换，但是不同的层之间参数是不同的。除了使用全连接层，你也可以使用卷积核为1的卷积网络。全连接层的输入输出都是$\ d_{model} = 512$ ，隐藏层的维度为$\ d_{ff} = 2048$ 。

### 3.4 Embeddings and Softmax

 &emsp;&emsp; 就像其他的序列转换模型那样，我们是用可学习的词向量来转化输入token和输出token，维度为$\ d_{\text{model}}$ ，我们也使用了可学习的线性变换和softmax函数来将decoder的输出转换成要预测的下一个token的概率。在我们的模型中，在两个embedding层和softmax、线性变换层中，我们共享权重矩阵（词向量矩阵），就像[24]那样。在embedding层中，我们会用$\ \sqrt{d_{\text{model}}}$ 乘上那些权重。

### 3.5 Positional Encoding

 &emsp;&emsp; 由于我们的模型不包括RNN和CNN，为了让模型利用序列的顺序，我们必须注入一些序列中token的相对或绝对位置的信息。出于这个目的，我们在encoder和decoder层的底部中的input embedding中增加了positional encodings。positional encodings也是有着$\ d_{\text{model}}$ 维度的向量，像词向量一样，因此，我们可以将两者相加，对于positoncal encoding而言，我们有许多选择，可以是可学习的，也可以是固定的。

 &emsp;&emsp; 在我们的工作中，我们使用不同频率的正弦和余弦函数：
$$
\begin{equation}
\begin{split}
PE_{(pos,2i)} &= sin(pos/10000^{2i/d_{\text{model}}})\\
PE_{(pos,2i+1)} &= cos(pos/10000^{2i/d_{\text{model}}})\\
\end{split}
\end{equation}
$$
 &emsp;&emsp; 其中$\ pos$ 表示位置，$\ i$ 表示维度，即$\ d_{\text{model}}$ 中的第几维，也就是说，positional encoding的没一个维度都对应一个正旋信号。波长以指数形式从$\ 2\pi$ 增长到$\ 10000·2\pi$ 。我们选择这个函数是因为，我们假设它允许模型很容易的学习到相对位置信息，因为，对于任何一个固定的位置偏移$\ k$ ，即$\ PE_{pos+k}$ ，我们都可以用$\ PE_{pos}$ 的线性函数表示出来。

 &emsp;&emsp; 我们也尝试了可学习的positional embedding[8]，发现两种方法得到了几乎一样的结果，而我们的正旋方法能够使得模型探索更长的序列长度，所以我们选择了我们的正旋方法。

## 4、Why Self-Attention

 &emsp;&emsp; 在这一节中，我们将对self-attetion层的各个方面与RNN和CNN进行比较，为了充分的利用我们的self-attention，我们将考虑三个方面。

 &emsp;&emsp; 一个是每一层的计算复杂度，另一个是可以并行化的计算量，通过所需的最小序列操作数来度量。

 &emsp;&emsp; 第三个是网络中远程依赖关系的路径长度。学习长距离的依赖是许多序列转换任务的关键挑战。影响这个关键因素的质量的因素是学习依赖时必须在网络中经历的前向和反向的路径长度。输入和输出序列中任何位置组合之间的路径越短，就越容易学习远程依赖[11]。因此，我们也在不同的层类型下比较了网络中任意两个输入和输出位置之间的最大路径长度。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/3.png?raw=true)

 &emsp;&emsp; 就像表1中现实的那样，self-attention使用了常数级别的序列操作数联系了所有的位置，但是RNN需要花费$\ O(n)$ 个Sequential operations。在时间复杂度方面，当$\ d<n$ 时，self-attention更快，这在word-piece，byte-pair等是非常常见的情况。为了提升长序列情况下的性能，我们可以使得我们的self-attention仅仅考虑$\ r$ 个上下文，这样，我们的最大路径长度会提升到$\ O(n/r)$ 。

 &emsp;&emsp; 另外，self-attention可以产生更多可解释的模型

## 5、Training

### 5.1 Training Data and Batching

 &emsp;&emsp; 我们在标准的WMT 2014 English-German数据集（翻译）上进行训练，数据集包含450万个句子对。句子是采用byte-pair编码的，他有一个共享的source-target表，包含37000个token。对于 English-French，我们使用了更大的WMT 2014 English-French数据集，包含36M个句子，将token分割成了32000个word-piece的词汇表。 句子对按近似的序列长度成批排列在一起。每个batch包含一组句子对，包含大约25000个source token和25000个target token。

### 5.2 Hardware and Schedule

 &emsp;&emsp; 采用了8块P100，每个训练步骤花费0.4秒。我们训练我们的base模型了100000个step，花费了12个小时，在我们的更大的模型上，每个step花费1秒，总共训练了300000个step，花费了3.5天。

### 5.3 Optimizer

 &emsp;&emsp; 使用Adam优化器，其中$\ \beta_1 = 0.9,\beta_2 = 0.98, \epsilon = 10^{-9}$ ，我们会动态的调整我们的学习率，其公式如下：
$$
lrate = d_{model}^{-0.5}·min(step\_num^{-0.5},step\_num·warmup\_steps^{-1.5})
$$
 &emsp;&emsp; 在最开始的$\ warmup\_steps$ 中，我们会线性的增加我们的学习率，然后按比例减少到step的平方根的倒数。我们采用$\ warmup\_steps = 4000$ 。

### 5.4 Regularization

#### 5.4.1 Residual Dropout

 &emsp;&emsp; 在进行残差链接和normalize之前，我们在每个子层的输出部分采用了dropout，其中$\ P_{drop} = 0.1$ 。

#### 5.4.2 Label Smoothing

 &emsp;&emsp; 采用了label smoothing，其中$\ \epsilon_{ls} = 0.1$ 。这降低了复杂性，但使得模型学习到了不确定性，提升了准确率和BLEU分数。



## 6、Results

### 6.1 Machine Translation

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/4.png?raw=true)

### 6.2 Model Variations

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89Transformer/5.png?raw=true)

## 7、Conclusion

 &emsp;&emsp; 在本文中，我们提出了Transformer，第一个仅仅基于attention的序列转换模型，使用multi-header self-attention取代了最常见的RNN层。

 &emsp;&emsp; 在机器翻译任务中，Transformer的训练速度比基于RNN和CNN的模型块的多。在两个任务中都取得了SOTA的结果。

 &emsp;&emsp; 我们对基于注意力的模型的未来感到兴奋，并计划将它们应用到其他任务中。我们计划将Transformer扩展到涉及文本以外的输入和输出模式的问题，并研究本地受限的注意力机制，以有效处理大型输入和输出，如图像、音频和视频。减少一代人的连续性是我们的另一个研究目标。