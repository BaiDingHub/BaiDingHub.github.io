---
title: NLP词向量篇（三）FastText
date: 2020-09-02 05:20:00
tags:
 - [深度学习]
 - [NLP基础知识]
categories: 
 - [深度学习,NLP基础知识]
keyword: "深度学习,自然语言处理，词向量"
description: "NLP词向量篇（三）FastText"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



#  Bag of Tricks for Efficient Text Classification  
> 时间：2016年
>
> 关键词：NLP, Word Embedding
>
> 论文位置：https://arxiv.org/pdf/1607.01759
>
> 引用：Joulin A, Grave E, Bojanowski P, et al. Bag of tricks for efficient text classification[J]. arXiv preprint arXiv:1607.01759, 2016.

**摘要：**本文探讨了一种简单有效的用于文本分类的baseline。实验表明，**fastText分类器在准确率方面与深度学习分类器相当，在训练和评估方面比深度学习分类器快很多个数量级。**使用标准的多核CPU，我们可以在不到10分钟的时间内对fastText进行10亿个单词的训练，并在不到一分钟的时间内对312K个类中的50万个句子进行分类

**索引**- 自然语言处理，词向量

## 预备知识

### 什么是N-gram

该部分转自https://zhuanlan.zhihu.com/p/32965521

 &emsp;&emsp; 在文本特征提取中，常常能看到N-gram的身影。它是一种基于语言模型的算法，基本思想是将文本内容按照字节顺序进行大小为N的滑动窗口操作，最终形成长度为N的字节片段序列。看下面的例子：

 &emsp;&emsp; 我来到达观数据参观

 &emsp;&emsp; 相应的bigram特征为：我来 来到 到达 达观 观数 数据 据参 参观

 &emsp;&emsp; 相应的trigram特征为：我来到 来到达 到达观 达观数 观数据 数据参 据参观

 &emsp;&emsp; 注意一点：N-gram中的gram根据**粒度不同，有不同的含义**。**它可以是字粒度、词粒度，也可以是字符粒度的**。

 &emsp;&emsp; 上面所举的例子属于**字粒度**的n-gram，**词粒度**的n-gram看下面例子：

 &emsp;&emsp; 我 来到 达观数据 参观

 &emsp;&emsp; 相应的bigram特征为：我/来到 来到/达观数据 达观数据/参观

 &emsp;&emsp; 相应的trigram特征为：我/来到/达观数据 来到/达观数据/参观

 &emsp;&emsp; **字符粒度的N-gram是用来表示一个单词**的，用字符粒度的N-gram来表示单词"apple"，设其超参数$\ n=3$ ，则以每个字符作为中心，得到其trigram为：

```
 “<ap”, “app”, “ppl”, “ple”, “le>”
```

 &emsp;&emsp; 其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“apple”这个单词，进一步，我们可以用这5个trigram的向量叠加来表示“apple”的词向量。

 &emsp;&emsp; 字符粒度的N-gram带来两点**好处**：

1. 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。

2. 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。

 &emsp;&emsp; N-gram产生的特征只是作为文本特征的候选集，你后面可能会采用信息熵、卡方统计、IDF等文本特征选择方式筛选出比较重要特征。



## 内容分析

 &emsp;&emsp; fastText这篇论文主要是讲解了一些在文本分类任务中的trick，来加速模型的训练，有些任务能够加速特别特别多，主要用到的trick有以下，两个字符级N-gram的引入和分层Sofrmax分类的使用：

### Hierarchical softmax

 &emsp;&emsp; **使用了基于Huffman编码树的hierarchical softmax**  (Goodman, 2001) (Mikolov et al.， 2013)。在训练过程中，计算复杂度$\ O(hk)$ 降至$\ O(hlog_2(k))$ ，其中$\ k$ 表示分类数目，$\ h$ 表示词向量的维度。

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/1.png?raw=true)

### N-grams features

 &emsp;&emsp; 针对一个句子而言，其作为模型的输入时，我们会将其每一个单词作为一个输入特征$\ x_i$ ，该输入特征是指的他的词向量，同时，我们要得到该词的字符级N-grams特征，作为该特征的附加特征，一同输入到模型中。



### 模型

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/2.png?raw=true)

 &emsp;&emsp; 模型与(Mikolov et al.， 2013)的CBOW十分类似，都包含输入层、隐藏层和输出层，同时CBOW也考虑了分层Softmax，唯一的不同就是字符级N-gram的引入。

 &emsp;&emsp; 以文本分类为例，fastText是一个监督学习任务，其词向量作为模型的隐式参数而被学习到。对于一个文本，我们会将其每一个单词作为一个输入特征$\ x_i$ ，该输入特征是指的他的词向量，同时，我们要得到该词的字符级N-grams特征，作为该特征的附加特征，一同输入到模型中。

 &emsp;&emsp; 输入层到隐藏层这一部分的作用是生成用来表征文本的向量，在fastText中，将所有的词向量以及字符级N-gram词向量取平均，得到的就是表征该文本的向量，该取平均的思想就是词袋法。之后将该向量送到隐藏层，之后通过分层Softmax进行分类。

### 为什么fastText如此快，效果还挺好？

 &emsp;&emsp; 假设我们有两段文本：

 &emsp;&emsp; 我 来到 达观数据

 &emsp;&emsp; 俺 去了 达而观信息科技

 &emsp;&emsp; 这两段文本意思几乎一模一样，如果要分类，肯定要分到同一个类中去。但在传统的分类器中，用来表征这两段文本的向量可能差距非常大。传统的文本分类中，你需要计算出每个词的权重，比如TFIDF值， “我”和“俺” 算出的TFIDF值相差可能会比较大，其它词类似，于是，VSM（向量空间模型）中用来表征这两段文本的文本向量差别可能比较大。但是fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度，于是，在fastText模型中，这两段文本的向量应该是非常相似的，于是，它们很大概率会被分到同一个类中。

 &emsp;&emsp; **使用词embedding而非词本身作为特征**，这是fastText效果好的一个原因；另一个原因就是**字符级n-gram特征的引入对分类效果会有一些提升** 。



## 1、Intorduction

 &emsp;&emsp; 文本分类是自然语言处理中的一项重要任务，应用广泛，如web搜索、信息检索、排序和文档分类(Deerwester et al., 1990;Pang and Lee, 2008)。最近，基于神经网络的模型越来越受欢迎(Kim, 2014; Zhang and LeCun, 2015;Conneau et al., 2016)。虽然这些模型在实践中取得了非常好的性能，但它们在训练和测试时往往相对较慢，这限制了它们在非常大的数据集上的使用。

 &emsp;&emsp; 同时，线性分类器通常被认为是文本分类问题中比较强的baseline(Joachims, 1998;McCallum and Nigam, 1998; Fan et al., 2008)。尽管它们很简单，但如果使用了正确的特征，它们往往能获得SOTA的表现(Wang和Manning,  2012)。它们还具有拓展到非常大的语料库的潜力(Agarwal等人，2014)。

 &emsp;&emsp; 在本研究中，我们探索了在文本分类的背景下，如何将这些baseline扩展到具有大输出空间的超大语料库。受最近高效的词向量学习工作的启发(Mikolov et al., 2013; Levy et al., 2015))，我们证明，带有秩约束和快速损失近似的线性模型可以在十分钟内训练10亿个单词，同时实现SOTA的性能。我们在两个不同的任务上评估我们的fasttext1方法的质量，即标签预测和情绪分析。

## 2、Model architecture

 &emsp;&emsp; 一个简单而有效的句子分类的baseline是将句子表示为bag of words(BoW)，然后训练一个线性分类器，例如logistic回归或SVM (Joachims, 1998;Fan et al., 2008)。然而，线性分类器不共享特征和类之间的参数。这可能限制了它们在大输出空间情况下的泛化，因为有些类只有很少的示例。通常的解决方法是将线性分类器分解为低秩矩阵(Schutze, 1992;Mikolov et al., 2013)或使用多层神经网络(Collobert and Weston, 2008;Zhang et al., 2015)。

 &emsp;&emsp; 图1显示了一个带有秩约束的简单线性模型。第一个权重矩阵A是单词的查找表。然后将词向量取平均得到一个文本表示，然后将文本表示送给线性分类器。文本表示是一个潜在的可重用的隐藏变量。这种架构类似于Mikolov等人(2013)的CBOW模型（中间的单词被一个标签所替代）。我们使用softmax函数$\ f$ 来计算预先定义类别的概率分布。对于一组N个文档，这会使类的负对数似然值最小化：
$$
-\frac{1}{N} \sum_{n=1}^N y_n \log(f(BAx_n))
$$
 &emsp;&emsp; 其中$\ x_n$ 是将第n个文本特征规范化后的词向量，$\ y_n$ 是标签，$\ A,B$ 为权重矩阵。该模型采用随机梯度下降和线性衰减学习率在多个cpu上进行异步训练。

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/2.png?raw=true)

### 2.1 Hierarchical softmax

 &emsp;&emsp; 当类的数量很大时，计算线性分类器的计算开销很大。更准确地说，计算复杂度为$\ O(kh)$ ，其中$\ k$ 为类的数量，$\ h$ 为词向量的维数。为了提高我们的运行时间，我们**使用了基于Huffman编码树的hierarchical softmax**  (Goodman, 2001) (Mikolov et al.， 2013)。在训练过程中，计算复杂度降至$\ O(hlog_2(k))$ 。

 &emsp;&emsp; 在寻找最接近的类时，hierarchical softmax在测试阶段也是有益的。Huffman编码树的每个节点都与一个概率相关联，叶子节点的概率就是从根节点到该节点的概率乘积。如果深度为$\ l+1$ 处节点的父节点为$\ n_1,...,n_l$ ，那么他的概率值为：
$$
P(n_{l+1}) = \prod_{i=1}^l P(n_i)
$$
 &emsp;&emsp; 这意味着节点的概率总是低于其父节点的概率。对树进行深度优先搜索并跟踪叶节点之间的最大概率，这样我们就可以丢弃任何概率很小的分支。在实践中，我们观察到复杂度在测试时降低到$\ O(hlog2(k))$ 。这种方法被进一步扩展到使用二进制堆以$\ O(log(T))$ 为代价计算T-top目标。

###  2.2 N-gram features

 &emsp;&emsp; 词袋的方法不会考虑词的顺序，但是简单的考虑词的顺序会使得计算代价非常大。**我们使用一个包含n个字母的包作为附加特性，以捕获关于局部词序的部分信息**。这在实践中是非常有效的，可以获与显式使用顺序的方法相当的结果(Wang和Manning, 2012)。

 &emsp;&emsp; 通过使用hash技巧 (Weinberger et al., 2009)（与（Mikolov et al. (2011）中的技巧相同），我们保持对N-grams快速和高效的内存映射。如果使用bigram只需要10M的bins，否则需要100M的bins。

## 3、Experiments

 &emsp;&emsp; 我们在两个不同的任务上评估fastText。首先，我们将其与现有的文本分类器在情感分析问题上进行了比较。然后，我们评估它在标签预测数据集中拓展到大输出空间的能力。注意我们的模型可以用Vowpal Wabbit库来实现，但是我们在实践中观察到，我们的定制实现至少要快2-5倍。

### 3.1 Sentiment analysis

#### 3.1.1 Datasets  and  baselines

 &emsp;&emsp; 我们采用了（Zhang et al. (2015））中使用过的8个数据集以及评价策略，采用了n-grams和TFIDF baseline与character level convolutional model (char-CNN) of Zhang and LeCun (2015)，character based convolution recurrent network (char-CRNN) of (Xiao and Cho, 2016) ，very deep convolutional network (VDCNN) of Conneau et al. (2016)。我们还采用了（Tang et al. (2015) ）的评价方法，使用了他们的两个主要baseline，Conv-GRNN and LSTM-GRNN。

#### 3.1.2 Results

 &emsp;&emsp; 我们在表1中展示了结果。我们使用10个隐藏神经元，在验证集上运行fastText 5个epoch。从{0.05,0.1,0.25,0.5}中选择的验证集的学习速率运行5个epoch的fastText。在这个任务中，添加三元信息可以提升1  -  4%的性能。总的来说，我们的准确率比char-CNN和char-CRNN稍好，比VDCNN稍差。注意，我们可以通过使用更多的n-grams略微提高精度，例如使用三元组，搜狗的性能上升到97.1%。最后，图3显示了我们的方法与Tang  et al.(2015)提出的方法是有竞争力的。我们调优了验证集上的超参数，并观察到使用n-grams到5可以获得最佳性能。与Tang et  al.(2015)不同，fastText没有使用预先训练好的词向量，这可以解释1%的准确率差

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/3.png?raw=true)

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/4.png?raw=true)

#### 3.1.3 Tranning time

 &emsp;&emsp; char-Cnn和VDCNN都是在NVIDIA Tesla K40  GPU上训练的，而我们的模型是在使用20个线程的CPU上训练的。表2显示了使用卷积的方法比fastText慢几个数量级。虽然通过使用最近CUDA的卷积实现，可以为char-CNN提速10倍，但是fastText只需要不到一分钟的时间就能在这些数据集上进行训练。Tang等人(2015)的GRNNs方法在CPU上使用单线程时，每个epoch大约需要12小时。相比于那些神经网络模型，我们的模型可以提高15000倍的速度。

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/5.png?raw=true)

### 3.2 Tag prediction

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/6.png?raw=true)

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/NLP%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/NLP%E8%AF%8D%E5%90%91%E9%87%8F%E7%AF%87%EF%BC%88%E4%B8%89%EF%BC%89FastText/7.png?raw=true)

## 4、Discussion and conclusion

 &emsp;&emsp; 在本文中，我们提出了一种简单的文本分类方法的baseline。与来自word2vec的未经监督训练的词向量不同，我们的单词特征可以平均在一起形成好的句子表征。在一些任务中，fastText的性能与最近提出的基于深度学习的方法相当，但速度要快得多。尽管从理论上讲，深度神经网络比浅层模型具有更高的代表性，但像情绪分析这样的简单文本分类问题是否适合用来评估它们还不清楚。我们将发布我们的代码，以便研究社区可以轻松地在我们的工作基础上进行构建。