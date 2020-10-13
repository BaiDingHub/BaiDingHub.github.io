---
title: NLP词向量篇（五）GPT-1
date: 2020-09-21 15:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（五）GPT-1"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# Improving Language Understanding by Generative Pre-Training
> 时间：2018年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf
>
> 引用：Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.

**摘要：**自然语言理解包括各种各样的任务，如文本蕴涵、问题回答、语义相似度评估和文档分类。虽然**大量的未标记文本语料库是丰富的，但用于学习这些特定任务的标记数据是稀缺的，这使得区分训练的模型很难充分执行**。我们证明，在这些任务中，**可以通过生成语言模型对不同的未标记文本语料库进行预训练，然后对每个特定任务进行有区别的微调来实现**。与之前的方法相比，**我们在微调期间使用任务感知的输入转换，从而在对模型体系结构进行最小更改的同时实现有效的迁移**。我们在自然语言理解的一系列baseline测试中证明了我们的方法的有效性。我们的模型（在未知下游任务时）比那些使用为每个任务专门设计的架构的经过区别训练的模型表现更好，在研究的12个任务中有9个任务的SOTA得到了显著改善。例如，我们在常识推理(故事完形填空测试)上取得了8.9%的绝对进步，在回答问题(种族)上取得了5.7%的绝对进步，在文本蕴涵(多项)上取得了1.5%的绝对进步。

**索引**- 自然语言处理，动态词向量

## 内容分析

 &emsp;&emsp; GPT可以说是ELMo的进化版本，其与ELMo的不同有：

- 使用了**特征提取能力更强的Transformer**来进行特征提取
- 不同的NLP任务的输入是不一样的，比如文本相似性任务是给两个句子，而文本分类任务只给一个句子。**GPT将不同任务的输入进行了转换，转换成了同一类型的输入**。这样避免了我们为每一种NLP任务设计不同的架构
- **添加了辅助目标函数来辅助fine-tuning的过程**

 &emsp;&emsp; GPT主要干了三件事

1. 利用Transformer设计一个representation表示模型，使得每个输入的token都能够得到一个比较好的representation，之后在一个规模比较大的训练集上对其进行预训练，这个数据集可能包含有多种不同的任务，但都是非监督任务。
2. Fine-tunning，即在一个下游任务（监督任务）中，加入预训练的模型，进行fine-tunning来拟合这个下游任务。到这，也就是GPT模型的整个过程就完成了，下面对其进行具体分析。
3. 对不同的下游任务进行了统一化，即将不同类型的输入转换成同一类型的输入，大大简化了架构设计这一步骤。

### 1）pre-training阶段

 &emsp;&emsp; 在预训练阶段，给定语料，$\ \mathcal{U} = \{u_1,...,u_n\}$ ，我们的目标是根据某个词的上下文预测这个词，即目标函数为：
$$
L_1(\mathcal{U}) = \sum_i \log P(u_1|u_{i-k},...,u_{i-1};\Theta) \tag{1}
$$
 &emsp;&emsp; 其中$\ k$ 表示上下文窗口的大小， 我们使用参数为$\ \Theta$ 的神经网络建模条件概率$\ P$ ，我们使用梯度下降来训练这些参数。

 &emsp;&emsp; 作者**使用了多层Transformer的decoder层**来建模$\ P$ 。对上下文采取了multi-head attention、self-attention、mask操作，之后使用position-wise的前馈层来生成目标token输出的分布，即：
$$
\begin{equation}
\begin{split}
h_0 &= UW_e + W_p\\
h_l &= \text{transformer_block}(h_{l-1}) \ \forall i \in [1,n]\\ 
P(u) &= \text{softmax}(h_n W_e^T)
\end{split}
\end{equation}
\tag{2}
$$
 &emsp;&emsp; 其中，$\ U = (u_{i-k},...,u_{i-1})$ ，是token的上下文向量，$\ n$ 使transformer层的数量，$\ W_e$ 是词向量矩阵，$\ W_p$ 是positon embedding矩阵。

 &emsp;&emsp; 可以看到，在这个模型中，**使用了Transformer的mask操作，也就意味着GPT使用的是单向Transformer，这是GPT的限制，在Bert中对这个问题进行了改进。**

### 2）fine-tuning阶段

 &emsp;&emsp; 上面训练完模型后，我们就把预训练的模型以及参数融入到下游任务中。对于某个下游任务，设其数据集$\ \mathcal{C}$ （带标签），某个输入的token为$\ (x^1,...,x^m)$ ， 对应的标签为$\ y$ 。将该输入送入我们的预训练模型中得到Transformer的输出$\ h_l^m$ 。然后将其送入到一个附加线性输出层中，该层的参数为$\ W_y$ ，来预测$\ y$ ：
$$
P(y|x^1,...,x^m) = \text{softmax}(h_l^m W_y) \tag{3}
$$
 &emsp;&emsp; 那么，在该任务中的目标函数就是最大化下面的目标函数：
$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y|x^1,...,x^m) \tag{4}
$$
 &emsp;&emsp; 另外，我们发现**将语言模型的目标函数作为一个辅助目标函数来fine-tunning，是有帮助的，可以提升监督模型的泛化性并加速收敛。**所以，我们的优化函数变成了：
$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C}) \tag{5}
$$
 &emsp;&emsp; 总的来说，在fine-tunning期间，我们唯一需要调整的参数就是$\ W_y$ 以及分隔符$\ \$$ 的词向量（见3.3节）。

### 3）下游任务的输入转换

 &emsp;&emsp; 对于某些任务，如文本分类，我们可以如上所述直接fine-tunning我们的模型。某些其他任务，如问题回答或文本蕴涵，具有结构化输入，如有序句子对，或文档、问题和答案的三元组。由**于我们的预训练模型是在连续的文本序列上训练的，因此需要进行一些修改才能将其应用到这些任务中**。我们使用 traversal-style的方法[52]，其中我们将结构化的输入转换为一个有序的序列，以便我们预先训练过的模型可以处理**。这些输入转换允许我们避免跨任务对架构进行大量更改。**我们在下面提供了这些输入转换的简要描述，图1提供了一个可视化的说明。所有的转换都包括添加随机初始化的开始和结束标记$\ (<s>,<e>)$。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/1.png?raw=true)

**Textual entailment**

 &emsp;&emsp; 对于蕴含任务来说，我们把premise $\ p$ 和hypothesis $\ h$ 合起来，成为一个token句子，并使用分隔符token$\ \$ $ 来分割。

**Similarity**

 &emsp;&emsp; 对于相似性任务来说，一组输入中的两个句子是没有固定的先后顺序的。为了反映这一点，我们修改输入序列，使其包含两种可能的句子顺序(中间有一个分隔符)，并分别处理它们以生成两个序列表示$\ h_l^m$ ，这两个表示在输入到线性输出层之前按元素添加。

**Question Answering and Commonsense Reasoning**

 &emsp;&emsp; 对于这些任务来说，给定一个语境文档$\ z$ ，一个问题$\ q$ ，一组可能的答案$\ \{a_k\}$ 。我们会把语境文档和问题与每一个可能的答案连起来，并在中间添加一个分隔符，如：$\ [z；q;\$;a_k]$ 。每一个这样的句子都会被我们的模型单独的处理，然后通过softmax层进行规范化，产生一个可能答案的输出分布。

### 4） GPT的效果

 &emsp;&emsp; GPT在12个任务中的9个都达到了SOTA的水平。看第四节



## 1、Introduction

 &emsp;&emsp; 有效地从原始文本中学习的能力对于减轻自然语言处理(NLP)中对监督学习的依赖至关重要。大多数深度学习方法需要大量手工标记的数据，这限制了它们在许多缺乏注释资源的领域中的适用性[61]。在这些情况下，能够从未标记数据中利用语言信息的模型为收集更多注释提供了一种有价值的替代方案，后者可能耗时且昂贵。此外，即使在相当大的监督是可用的情况下，用无监督的方式学习良好的representation可以提供一个显著的性能提升。到目前为止，最令人信服的证据是广泛使用预先训练过的词向量来提高一系列NLP任务的表现[8,11,26,45]。

 &emsp;&emsp; 然而，**从未标记的文本中利用不仅仅是word级别的信息是有挑战性的，主要有两个原因**。首先，**还不清楚在学习对迁移有用的文本representation时，哪种类型的优化目标是最有效的**。最近的研究着眼于各种目标，如语言建模[44]、机器翻译[38]和语篇连贯[22]，每种方法在不同任务上都优于其他方法。第二，**对于将学习到的representation转移到目标任务的最有效的方式没有共识**。现有的技术包括对模型体系结构进行任务特定变更的组合[44,44]，使用复杂的学习方案[21]和添加辅助学习目标[50]。这些不确定性使得开发有效的语言处理半监督学习方法变得困难。

 &emsp;&emsp; 在本文中，我们探索了一种语言理解任务的半监督方法，**使用无监督任务预训练和监督任务微调的组合**。**我们的目标是学习一种通用的表示法，这种表示法可以在不加调整的情况下，适应广泛的任务**。我们假设可以访问大量未标记文本的语料库和一些带有人工注释的训练示例(目标任务)的数据集。我们的setup不要求这些目标任务与未标记的语料库在同一个域中。**我们采用两阶段的训练步骤。首先，我们对未标记数据使用语言建模目标来学习神经网络模型的初始参数。随后，我们使用相应的监督目标将这些参数适应于目标任务。**

 &emsp;&emsp; 对于我们的模型架构，我们**使用了Transformer**[62]，它已经被证明可以在各种任务上出色地执行，如机器翻译[62]、文档生成[34]和语法解析[29]。与循环网络等替代方案相比，**这种模型选择为我们提供了一种更结构化的记忆，用于处理文本中的长期依赖关系，从而在不同任务中实现了稳健的传输性能**。**在迁移过程中，我们利用从traversal-style方法[52]派生出来的的特定于任务的输入调整，它将结构化文本输入处理为单个连续的token序列。正如我们在实验中所演示的，这些调整使我们能够在对预训练模型的架构进行最小更改的情况下有效地进行微调**

 &emsp;&emsp; 我们在四种类型的语言理解任务上评估我们的方法——自然语言推理、问题回答、语义相似度和文本分类。我们的模型（在未知下游任务时）比那些为每个任务专门设计架构的经过区别训练的模型表现更好，在研究的12个任务中，有9个任务的技术水平得到了显著提高。例如，我们在常识推理(Stories Cloze Test)[40]上取得了8.9%的绝对进步，在回答问题(RACE)[30]上取得了5.7%的绝对进步，在文本蕴涵(MultiNLI)[66]上取得了1.5%的绝对进步，在最近引入的GLUE多任务基准测试[64]上取得了5.5%的绝对进步。我们还分析了预训练模型在四种不同环境下的zero-shot行为，证明它获得了对下游任务有用的语言知识。

## 2、Related Work

### 2.1 Semi-supervised learning for NLP

 &emsp;&emsp; 我们的工作基本上属于自然语言的半监督学习范畴。这种范式引起了人们的极大兴趣，应用于诸如序列标记[24,33,57]或文本分类[41,70]等任务。**最早的方法使用未标记的数据来计算word级或phrase级的统计，然后将其作为特征用于监督模型[33]中**。在过去的几年里，研究人员已经证明了使用**词向量**的好处[11,39,42]，**它是在未标记的语料库上训练的，可以提高在各种任务上的表现[8,11,26,45]**。然而，**这些方法主要传递word级的信息，而我们的目标是捕获更高级别的语义。**

 &emsp;&emsp; 最近的方法已经研究了学习和利用更多的词汇水平的语义从未标记的数据。phrase级或sentence级词向量可以使用未标记语料库进行训练，它们被用于将文本编码为适合各种目标任务的向量表示[28,32,1,36,22,12,56,31]。

### 2.2 Unsupervised pre-training

 &emsp;&emsp; 无监督预训练是半监督学习的一种特殊情况，其目标是找到一个好的初始化点，而不是修改有监督学习目标。早期的工作探索了该技术在图像分类中的使用[20,49,63]和回归任务[3]。后续[15]研究表明，预训练可以作为一种正则化方案，在深度神经网络中具有更好的泛化效果。在最近的工作中，该方法已被用于帮助训练深度神经网络完成各种任务，如图像分类[69]、语音识别[68]、实体消歧[17]和机器翻译[48]。

 &emsp;&emsp; 与我们最接近的工作是使用语言建模目标对神经网络进行预训练，然后在监督下对目标任务进行微调。Dai等[13]和Howard和Ruder[21]采用了这种方法来改进文本分类。然而，虽然与训练phase有助于捕获一些语言信息，但是他们的模型使用LSTM模型，这限制了模型的预测能力。相比之下，我们选择的Transformer网络允许我们捕获更大范围的语言结构，正如我们的实验所证明的那样。此外，我们还演示了我们的模型在更广泛的任务上的有效性，包括自然语言推断、释义检测和故事完成。其他方法[43,44,38]在训练目标任务上的监督模型时，使用来自2个预先训练语言或机器翻译模型的隐藏表征作为辅助特征。这涉及到每个单独目标任务的大量新参数，而在传输过程中我们需要对模型架构进行最小的更改。

### 2.3 Auxiliary training objectives

 &emsp;&emsp; 增加辅助的无监督训练目标是半监督学习的另一种形式。在早期的工作中，Collobert和Weston[10]使用了各种各样的辅助NLP任务，如词性标记、分块、命名实体识别和语言建模来改进语义角色标记任务的效果。最近，Rei[50]在他们的目标任务目标中增加了一个辅助语言建模目标，并演示了序列标记任务的性能的提高。我们的实验也使用了一个辅助目标，但正如我们所展示的，未经监督的预训练已经学会了与目标任务相关的几个语言方面。

## 3、Framework

 &emsp;&emsp; 我们的训练过程包括两个阶段。第一阶段是在大语料库上学习高容量的语言模型。接下来是一个微调阶段，在这个阶段中，我们会使模型适应这个具有标记数据的任务。

### 3.1 Unsupervised pre-training

 &emsp;&emsp; 给定一个非监督的语料，其token为$\ \mathcal{U} = \{u_1,...,u_n\}$ ，我们使用一个标准的语言建模目标，即最大化下面的似然函数：
$$
L_1(\mathcal{U}) = \sum_i \log P(u_1|u_{i-k},...,u_{i-1};\Theta) \tag{1}
$$
 &emsp;&emsp; 其中$\ k$ 表示上下文窗口的大小， 我们使用参数为$\ \Theta$ 的神经网络建模条件概率$\ P$ ，我们使用梯度下降来训练这些参数。

 &emsp;&emsp; 在我们的实验中，我们使用了多层Transformer decoder[34作为语言模型，这是一个transformer的变体[62]。这个模型对输入的上下文token采用了multi-headed self-attention操作，之后使用position-wise的前馈层来生成目标token输出的分布，即（这里只用了transformer的decoder）：
$$
\begin{equation}
\begin{split}
h_0 &= UW_e + W_p\\
h_l &= \text{transformer_block}(h_{l-1}) \ \forall i \in [1,n]\\ 
P(u) &= \text{softmax}(h_n W_e^T)
\end{split}
\end{equation}
\tag{2}
$$
 &emsp;&emsp; 其中，$\ U = (u_{i-k},...,u_{i-1})$ ，使token的上下文向量，$\ n$ 使transformer层的数量，$\ W_e$ 是词向量矩阵，$\ W_p$ 是positon embedding矩阵。

### 3.2 Supervised fine-tuning

 &emsp;&emsp; 将公式1作为目标函数训练完模型后，我们就把参数应用到监督任务中，我们假设，带标签的数据集$\ \mathcal{C}$ ，在数据集中，每个实例都包含一个句子，其token为$\ x^1,...,x^m$ ，对应的标签为$\ y$ 。将该输入送入我们的预训练模型中得到transformer模块的输出$\ h_l^m$ ，然后将其送入到一个附加线性输出层中，该层的参数为$\ W_y$ ，来预测$\ y$ ：
$$
P(y|x^1,...,x^m) = \text{softmax}(h_l^m W_y) \tag{3}
$$
 &emsp;&emsp; 那么，在该任务中的目标函数就是最大化下面的目标函数：
$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y|x^1,...,x^m) \tag{4}
$$
 &emsp;&emsp; 另外，我们发现**将语言模型的目标函数作为一个辅助目标函数来fine-tunning，是有帮助的，可以提升监督模型的泛化性并加速收敛。**所以，我们的优化函数变成了：
$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C}) \tag{5}
$$
 &emsp;&emsp; 总的来说，在fine-tunning期间，我们唯一需要调整的参数就是$\ W_y$ 以及分隔符$\ \$$ 的词向量（见3.3节）。

### 3.3 Task-specific input transformations

 &emsp;&emsp; 对于某些任务，如文本分类，我们可以如上所述直接fine-tunning我们的模型。某些其他任务，如问题回答或文本蕴涵，具有结构化输入，如有序句子对，或文档、问题和答案的三元组。由**于我们的预训练模型是在连续的文本序列上训练的，因此需要进行一些修改才能将其应用到这些任务中**。之前的工作提出了transferred representations[44]之上的特定学习任务架构。这种方法重新引入了大量特定于任务的定制架构组件，但并且没有为这些附加的架构组件使用迁移学习。相反，我们使用 traversal-style的方法[52]，其中我们将结构化的输入转换为一个有序的序列，以便我们预先训练过的模型可以处理**。这些输入转换允许我们避免跨任务对架构进行大量更改。**我们在下面提供了这些输入转换的简要描述，图1提供了一个可视化的说明。所有的转换都包括添加随机初始化的开始和结束标记$\ (<s>,<e>)$。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/1.png?raw=true)

**Textual entailment**

 &emsp;&emsp; 对于蕴含任务来说，我们把premise $\ p$ 和hypothesis $\ h$ 合起来，成为一个token句子，并使用分隔符token$\ \$ $ 来分割。

**Similarity**

 &emsp;&emsp; 对于相似性任务来说，一组输入中的两个句子是没有固定的先后顺序的。为了反映这一点，我们修改输入序列，使其包含两种可能的句子顺序(中间有一个分隔符)，并分别处理它们以生成两个序列表示$\ h_l^m$ ，这两个表示在输入到线性输出层之前按元素添加。

**Question Answering and Commonsense Reasoning**

 &emsp;&emsp; 对于这些任务来说，给定一个语境文档$\ z$ ，一个问题$\ q$ ，一组可能的答案$\ \{a_k\}$ 。我们会把语境文档和问题与每一个可能的答案连起来，并在中间添加一个分隔符，如：$\ [z；q;\$;a_k]$ 。每一个这样的句子都会被我们的模型单独的处理，然后通过softmax层进行规范化，产生一个可能答案的输出分布。



## 4、Experiments

### 4.1 Setup

#### 4.1.1 Unsupervised pre-training

 &emsp;&emsp; 我们使用BooksCorpus数据集[71]来训练语言模型。它包含了7000多本独特的未出版的书籍，包括各种类型的书，冒险、幻想和浪漫。至关重要的是，它包含了长序列的连续文本，这使得生成模型能够以远程信息为条件进行学习。另一种数据集，1B大小的Word Benchmark，被类似的方法使用，ELMo[44]，与BooksCorpus大约是相同的大小，但打乱了句子，破坏了远程结构。我们的语言模型在该语料库上实现了一个非常低的token界别的复杂度18.4。

#### 4.1.2 Model specifications

 &emsp;&emsp; 我们的模型很大程度上**遵循了Transformer的原始工作**[62]。我们训练了一个**12层的只有decoder的Transformer**，同时**使用了masked self-attention heads**(768 dimensional states and 12 attention heads)。对于 **position-wise的前馈网络，我们使用3072维的inner states**。我们使用**Adam优化方案**[27]，最大学习率为2.5e-4。**学习率在前2000个epoch线性增加，之后使用cosine schedule退火到0**。我们对由**512个token组成的64个随机抽样的连续序列进行100个epoch的训练**。由于在整个模型中广泛使用了layernorm[2]，因此$\ N(0,0.02)$ 的简单权重初始化就足够了。我们使用了40000个字节对编码(BPE)词汇表，使**用残差链接、embedding、0.1的dropout的正则化方法**。我们还**采用了[37]中提出的L2正则化的修改版本**，参数设置为$\ w  = 0.01$ 。对于激活函数，我们使用**Gaussian Error Linear Unit (GELU)**[18]。我们**使用了可学习的position embeddings 代替了原始工作中提出的正弦版本**。我们**使用ftfy  library来清理BooksCorpus中的原始文本，标准化一些标点符号和空格，并使用spaCy tokenizer**

#### 4.1.3 Fine-tuning details

 &emsp;&emsp; 我们还是使用的非监督与训练中的超参设定，在分类器上增加0.1比例的dropout，在大多数任务上，学习率设置为6.25e-5，batch设置为32。我们的模型fine-tunning的很快，只需要3个epoch。同时，我们使用了一个线性学习率衰减策略。$\ \lambda = 0.5$ 。

### 4.2 Supervised fine-tuning

 &emsp;&emsp; 我们在各种监督任务上进行实验，包括自然语言推理、问题回答、语义相似度和文本分类。其中一些任务可以在最近发布的GLUE多任务基准测试[64]中使用。图1提供了所有任务和数据集的概述。

#### 4.2.1 Natural Language Inference

 &emsp;&emsp; 自然语言推理(NLI)的任务，又称识别文本蕴涵，包括阅读一对句子，从蕴涵、矛盾或中性的一个句子来判断句子之间的关系。尽管最近有很多人对此感兴趣[58,35,44]，但由于词汇暗含、共指、词汇和句法歧义等现象的广泛存在，这项任务仍然具有挑战性。我们评估了五个不同来源的数据集，包括image captions (SNLI), transcribed speech, popular fiction, and government  reports (MNLI), Wikipedia articles (QNLI), science exams (SciTail) or news  articles (RTE).

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/2.png?raw=true)

#### 4.2.2 Question answering and commonsense reasoning

 &emsp;&emsp; 另一个需要单句和多句推理的任务是回答问题。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/3.png?raw=true)

#### 4.2.3 Semantic Similarity

 &emsp;&emsp; 语义相似度(或意译检测)任务包括预测两个句子在语义上是否等价。挑战在于认识概念的重新措辞、理解否定和处理句法歧义。

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/4.png?raw=true)

## 5、Analysis

### 5.1 Impact of number of layers transferred

 &emsp;&emsp; 我们观察了将可变数量的层从无监督的预训练迁移到有监督的目标任务的影响。图2(左)说明了我们的方法在MultiNLI和RACE上的性能，它与迁移的层数有关。我们观察到的标准结果是迁移embedding提高了性能，在MultiNLI.任务上，**每个Transformer层提供了可达9%的性能提升。这表明预训练的模型中的每一层都包含解决目标任务的有用功能。**

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/5.png?raw=true)

### 5.2 Zero-shot Behaviors

 &emsp;&emsp; **我们想更好地理解为什么Transformer的语言模型预训练是有效的。一种假设是，潜在的生成模型能够学着去执行我们评估的许多任务，以提高其语言建模能力，而与LSTMs相比，transformer更结构化的注意记忆有助于迁移。**我们设计了一系列启发式解决方案，使用潜在的生成模型来执行任务，而无需监督微调。我们在图2(右)中可视化了这些启发式解决方案在生成式预训练过程中的有效性。我们观察到，这些启发式的性能是稳定的，并且在训练过程中稳步增加，这表明生成式预训练支持了对各种任务相关功能的学习。我们还观察到，LSTM在其zero-shot性能方面表现出更高的方差，这表明Transformer结构归纳性的偏置有助于迁移。

 &emsp;&emsp; 对于CoLA(语言可接受性），用生成模型分配和预测的平均token日志概率对例子进行评分。对于SST-2（情感分析），我们在每一个例子中都非常附加了标记，并且将语言模型的输出分布限制为只有正和负两个词，并猜测它分配给更高概率的标记作为预测。对于race（question answering），我们选择生成模型在文档和问题条件下分配最高平均token日志概率的答案。对于DPRD[46]（winograd模式），我们用两个可能的引用来替换定代词，并预测生成模型在替换后将较高的平均token日志概率分配给序列的其余部分的分辨率。

### 5.3 Ablation studies

 &emsp;&emsp; 我们进行了三种不同的Ablation研究(表5)。首先，在微调过程中，我们检查了我们的方法在没有辅助LM目标的情况下的性能。我们观察到辅助目标对NLI任务和QQP有帮助。总的来说，**这一趋势表明，较大的数据集从辅助目标中受益，而较小的数据集则不能**。其次，通过与使用相同框架的单层2048单元LSTM进行比较，分析了Transformer的性能。我们观察到使用LSTM而不是Transformer时平均分数下降5.6。LSTM只在一个数据集MRPC上优于Transformer。最后，我们还将其与transformer架构进行了比较，我们直接在受监督的目标任务上进行了训练，而没有进行预训练。我们观察到，缺乏预训练会影响所有任务的表现，与我们的完整模型相比，结果导致14.8%的下降。

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%BA%94%EF%BC%89GPT-1/6.png?raw=true)

## 6、Conclusion

 &emsp;&emsp; 我们引入了一个框架，通过生成性预训练和区分性微调，以单一未知任务模型实现强大的自然语言理解。通过对不同语料库进行长距离连续文本的预先训练，我们的模型获得了重要的全局知识和处理长距离依赖关系的能力，然后成功地将这些知识迁移到解决特定性任务，如问答、语义相似性评估、限定确定和文本分类。改进了我们所研究的12个数据集中的9个的最新技术。利用无监督（预）训练提高识别任务的性能一直是机器学习研究的一个重要目标。我们的工作表明，实现显著的性能提升确实是可能的，并且提供了关于什么模型（(Transformers)）和数据集（具有长期依赖性的文本）最适合这种方法的提示。我们希望这将有助于对自然语言理解和其他领域的无监督学习进行新的研究，进一步提高我们对无监督学习如何以及何时起作用的理解。