---
title: NLP特征提取器篇（一）RNN
date: 2020-09-11 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，特征提取器"
description: "NLP特征提取器篇（一）RNN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89RNN/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



#  Recurrent Neural Network
> 介绍：本篇介绍NLP中最基础的特征提取器，RNN，同时，本篇中的内容引用了博客https://zybuluo.com/hanbingtao/note/541458
>
> 推荐阅读：[Recurrent Neural Network Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)



## 1、基本RNN

 &emsp;&emsp; 我们知道，RNN是专门用来处理序列信息的，在这些任务中，模型的输入往往是固定长度的序列（长度不够的话，用某些特定值填补），在下面的描述中，以词性标注任务为例：

**词性标注任务描述**

 &emsp;&emsp; 对于某个输入，例如，“我吃苹果”，这句话来说，我们的模型输出应该是“名词、动词、名词”这个序列。那么，我们是怎么样一步步把输入映射到输出空间的呢 ？

**分词**

 &emsp;&emsp; 对于输入句子来说，我们首先需要分词，将“我吃苹果”分成了“我  吃  苹果”，因此，RNN的模型输入应该是“我  吃  苹果”这个序列。

**词向量表示**

 &emsp;&emsp; 我们都知道，对于网络模型来说，不可能直接输入单词，因此，我们需要用一个向量来表示单词，可以是one-hot，也可以使用带有语义信息的词向量，比如，“我”这个单词，用one-hot表示为[0,1,0,0...]，因此，模型的输入也就转换成了这几个词向量的序列。当然，词向量的话，我们需要在模型训练前就训练好词向量（静态词向量），也可以输入为one-hot，在模型中动态的调整词向量（动态词向量）。

**真实的模型运行情况**

 &emsp;&emsp; 对于“我 吃 苹果”这个输入而言，我们需要先向模型输入“我”的词向量，得到模型的运行结果，然后输入“吃”的词向量，最后输入“苹果”的词向量，这就是RNN的序列化运行，但是我们都知道对于一个句子而言，一个词的上下文对该词的词性判断是有影响的，因此，在“吃”这个词进行运算时，我们需要用到“我”这个词在运算时得到的中间特征。因此，对于单个单词“吃”的输入运行情况来说，RNN的结构如下：

<img src="https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89RNN/1.png?raw=true" alt="1" style="zoom:50%;" />

 &emsp;&emsp; 其中，$\ X_t$ 代表输入向量，在这里表示“吃”的词向量，$\ U$ 代表输入层到隐藏层的权重矩阵，$\ S_t$ 代表隐藏层的向量值，正如我们上面所说的，我们需要用到上一个词的特征$\ S_{t-1}$ ，其中$\ W$ 代表$\ S_{t-1}$ 输入到该隐藏层的权重矩阵。$\ V$ 代表隐藏层到输出层的权重矩阵，$\ O_t$ 就是输出向量，在这里就代表“动词”。因此，在该情况下，其数学表示为：
$$
\begin{equation}
\begin{split}
S_t&=f(U·X_t + W·S_{t-1})\\
O_t&=g(V·S_t)\\
\end{split}
\end{equation}
$$
 &emsp;&emsp; 那么对于整个输入而言，他的模型运行情况是怎么样呢？其实，就是讲隐藏层的循环部分按时间线进行展开，那么其图形表示如下：

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89RNN/2.png?raw=true)

 &emsp;&emsp; 注意：这只是展开的示意图，所有的输入用的都是同样的隐藏层

## 2、双向RNN

 &emsp;&emsp; 我们会发现，基础RNN存在一个问题，那就是模型中只用到了上文的信息，但是没有用到下文的信息，这对语言模型来说是不太好的，因为下文的信息往往对这个词也有着非常重要的意义，因此，我们在设计模型时，需要把下文的特征加入到预测中去，因此，得到的模型示意图如下：

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89RNN/3.png?raw=true)

 &emsp;&emsp; 其中，蓝色的部分就是基础RNN的运行过程，在该基础之上，添加了橙色的流程。很明显，橙色部分的信息是从最后面传过来的，因此带有着下文的信息，也就解决了上面的问题。

 &emsp;&emsp; 那么，模型对应的数学表达式为：
$$
\begin{equation}
\begin{split}
S_t&=f(U·X_t + W·S_{t-1})\\
S'_t&=f(U'·X_t + W'·S_{t+1})\\
O_t&=g(V·S_t+V'S'_t)\\
\end{split}
\end{equation}
$$


## 3、深度RNN

 &emsp;&emsp; 前面介绍的RNN都只有一个隐藏层，在深度学习中，深度就是一切，因此，我们当然可以堆叠多个隐藏层，这样我们就得到了我们的循环神经网络

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89RNN/4.png?raw=true)

## 4、RNN的缺点

### 4.1 长期依赖问题

 &emsp;&emsp; 长期依赖产生的原因是当神经网络的节点经过许多阶段的计算后，之前比较长的时间片的特征已经被覆盖。

### 4.2 梯度爆炸和梯度消失问题

 &emsp;&emsp; 梯度消失和梯度爆炸是困扰RNN模型训练的关键原因之一，**产生梯度消失和梯度爆炸是由于RNN的权值矩阵循环相乘导致的，相同函数的多次组合会导致极端的非线性行为**。梯度消失和梯度爆炸主要存在RNN中，因为RNN中每个时间片使用相同的权值矩阵。对于一个DNN，虽然也涉及多个矩阵的相乘，但是通过精心设计权值的比例可以避免梯度消失和梯度爆炸的问题。

 &emsp;&emsp; 也因为梯度爆炸和梯度消失问题，使得RNN很难处理长序列问题。

 &emsp;&emsp; 梯度爆炸的解决方法：梯度裁剪（当梯度到达一定程度时，进行梯度裁剪）

 &emsp;&emsp; 梯度消失的解决方法：使用Relu函数替代Sigmoid和Tanh



## 5、CNN与RNN？

### 5.1 并行运算能力

 &emsp;&emsp; 对于一个输入而言，CNN可以并行的计算，而RNN由于某个词的运算需要用到上下文的信息，因此是串行的运算。