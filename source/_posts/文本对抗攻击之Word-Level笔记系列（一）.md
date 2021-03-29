---
title: 文本对抗攻击之Word-Level笔记系列（一）
date: 2021-03-01 05:20:00
tags:
 - [文本对抗攻击]
 - [Word-Level]
 - [论文笔记]
categories: 
 - [深度学习,文本对抗]
keyword: "深度学习,文本对抗,论文笔记"
description: "文本对抗攻击之Word-Level笔记系列（一）"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 一、Crafting adversarial input sequences for recurrent neural networks

## 1. Paper Information

> 时间：2016年
>
> 关键词：Adversarial Attack，White-box
>
> 论文位置：https://arxiv.org/pdf/1604.08275.pdf
>
> 引用：Papernot N, McDaniel P, Swami A, et al. Crafting adversarial input sequences for recurrent neural networks[C]//MILCOM 2016-2016 IEEE Military Communications Conference. IEEE, 2016: 49-54.

## 2. Motivation

 &emsp;&emsp; 此前，大多数的对抗样本大多处理CV的分类任务，在本篇论文中，我们将把对抗样本用于RNN网络中，用来处理序列数据。

## 3. Main Arguments

 &emsp;&emsp; 本篇论文形式化了序列数据下的对抗样本优化问题，针对RNN的特点，我们采用前向导数的构造算法。这包括演示如何计算循环计算图的前导数。我们研究了从模型的预处理输入到原始输入的对抗性扰动。我们使用RNN模型进行分类和顺序预测来评估我们的攻击方法的性能。平均而言，在71个单词的影评中改变9个单词足以让我们的分类RNN在对影评进行情绪分析时做出100%错误的类预测。

## 4. Framework

### 4.1 Adversarial Samples and Sequences

#### 4.1.1 Adversarial Samples

 &emsp;&emsp; 当输入是序列，但是输出是类别时，要解决的问题是：
$$
\vec{x^*} = \vec{x} + \delta_{\vec{x}} = \vec{x} + \min ||\vec{z}|| \ s.t. \ f(\vec{x} + \vec{z}) \ne f(\vec{x}) \tag{3}
$$

#### 4.1.2 Adversarial Sequences

 &emsp;&emsp; 当输入和输出都是序列时：
$$
\vec{x^*} = \vec{x} + \delta_{\vec{x}} = \vec{x} + \min ||\vec{z}|| \ s.t. ||\ f(\vec{x} + \vec{z})- \vec{y^*}|| \le \Delta \tag{4}
$$

### 4.2 Using FGSM

 &emsp;&emsp; 使用FGSM解决问题（3）：
$$
\vec{x^*} = \vec{x} + \delta_{\vec{x}} = \vec{x^*} = \vec{x} + \epsilon\ \text{sign}(\nabla_{\vec{x}} c(f,\vec{x}, \vec{y}))
$$
 &emsp;&emsp; 其中，$\ c$ 表示损失函数

### 4.3 Using Forward Derivative

 &emsp;&emsp; 前向导数是生成对抗样本的另一种方法，前向导数可以用模型的Jacobian矩阵定义：
$$
J_f[i,j] = \frac{\partial f_j}{\partial x_i}
$$
 &emsp;&emsp; 其中，$\ x_i$ 表示输入的第i个元素，$\ f_j$ 表示输出的第j个元素。它精确的估计了输出元素$\ f_j$ 对输入元素$\ x_i$ 的敏感性。

 &emsp;&emsp; 当计算图存在循环时，我们并不是那么容易计算前向导数，比如RNN模型。因此，作者使用了**计算图展开**技术。

 &emsp;&emsp; 那么，对于RNN来说，第t个时间步的输出是：
$$
h^{(t)}(\vec{x}) = \phi(h^{(t-1)}(\vec{x}), \vec{x}, \vec{w})
$$
 &emsp;&emsp; 将上述式子展开，我们得到：
$$
h^{(t)}(\vec{x}) = \phi(\phi(\dots\phi(h^{(t-1)}(\vec{x}), \vec{x}, \vec{w}),\dots  \vec{x}, \vec{w}), \vec{x}, \vec{w})
$$
 &emsp;&emsp; 我们通过展开他的递归分量，我们就使得RNN的计算图变成了无环的。这样，我们就可以计算前向梯度：
$$
J_f[i,j] = \frac{\partial f^{(j)}}{\partial x^{(i)}}
$$
 &emsp;&emsp; 其中，$\ x^{(i)}$ 表示第i步的输入序列，$\ y^{(j)}$ 表示第j步的输出序列。使用链式法则展开就可以计算。

 &emsp;&emsp; 因此，我们使用前向导数，我们既可以解决问题（3）也可以解决问题（4），即分类和序列问题。

 &emsp;&emsp; 在解决序列问题时，我们逐级的考虑输出序列。Jacobian矩阵的每一列对应于第j步的输出序列，如果第i步的输入序列在这一列有较高的绝对值，而其他列较小，说明该步在第j步的输出有很大影响。因此，我们只需要按照$\ \text{sign}(J_f[i,j]) \times \text{sign}(\vec{y_j^*})$ 这个方向去修改第i步的输入，我们就能够得到我们想要的输出（即分类为$\ \vec{y_j^*}$ ）

## 5.Result

### 5.1 Setup

 &emsp;&emsp; 作者使用了两种模型，一种是用于分类的RNN，用来分类影评的情感色彩，二分类任务，我们通过修改影评的词来误导分类器。另一种是用于序列输入和序列输出的RNN，基于雅可比矩阵的攻击通过识别每个输入序列步骤的贡献来改变模型的输出。

### 5.2 Recurrent Neural Networks with Categorical Output

 &emsp;&emsp; 采用了RNN影评分类器，我们在该模型上实现了100%的攻击成功率，在2000条影评（平均有71个词）上，平均只需要修改9.18个词。

 &emsp;&emsp; 作者的分类模型为：四层模型，输入层、LSTM层、Mean Pooling层和Softmax层：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/1.png?raw=true)

 &emsp;&emsp; 数据集包含2000条训练影评，500条测试影评。影评字典包含有10000个字。词向量的维度是128。模型在该数据集上实现了100%的训练准确率和78.21的测试准确率。

 &emsp;&emsp; 在攻击时，我们需要使用字典中的单词来修改输入句子，使得他的预测标签发生变化。首先，我们计算关于输入的词向量的Jacobian矩阵，即$\ J_f(\vec{x})[i,j] = \frac{\partial h_j}{\partial x_i}$ ，这为我们提供了，输入词向量与池化层输出的变化之间的映射关系。那么，对于输入序列的每个单词i，我们就可以根据Jacobian矩阵的值$\ \text{sign}(J_f(\vec{x})[i,f(\vec{x})]),其中f(\vec{x}) = \arg \max_{0,1}(p_j)$  得到每个词向量需要去扰动的方向。

 &emsp;&emsp; 而与CV中不同的是，我们面临了一个困难，即：**我们的词向量是有限的，即只有有限个词向量对应有词，因此，我们无法使得对抗样本中的词向量是任意的向量。** 为了解决这个问题，我们使用了下面的算法

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/2.png?raw=true)

 &emsp;&emsp; 我们从词典中，找到与$\ \text{sign}(J_f(\vec{x}[i,y]))$ 方向上最接近的词向量$\ \vec{z}$ 。通过这种方法，我们就找到了最接近Jacobian矩阵所表示的方向，也就是对模型预测影响最大的词向量。通过迭代的寻找，我们就可以得到我们的对抗样本。

### 5.3 Recurrent Neural Networks with Sequential Output

 &emsp;&emsp; 在该实验中，作者采用的是合成数据。我们对100个生成的输入和输出序列对进行训练。每步输入5个值，每步输出3个值。两个序列都有10步长。

![3](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/3.png?raw=true)

 &emsp;&emsp; 作者进行了简介，并没有详细描述。

## 6. Argument

 &emsp;&emsp; 该篇论文是第一篇文本领域的对抗样本研究，研究了在分类和序列输出情况下的攻击，但没有对序列输出进行详细描述，而且模型和数据集并没有那么正式。本文的方法属于白盒攻击。本文也告诉了我们，RNN模型也存在对抗样本的问题。

## 7. Further research

 &emsp;&emsp; 未来的工作还应该解决对抗性序列的语法问题，以提高它们的**语义意义**，并确保它们对人类是不可区分的。同时，应该要考虑**黑盒攻击**以及**迁移性问题**。





# 二、Towards Crafting Text Adversarial Samples.

## 1. Paper Information

> 时间：2017年
>
> 关键词：Adversarial Attack，White-box， Genre， Word-Level
>
> 论文位置：https://arxiv.org/pdf/1707.02812.pdf
>
> 引用：Samanta S, Mehta S. Towards crafting text adversarial samples[J]. arXiv preprint arXiv:1707.02812, 2017.

## 2. Motivation

## 3. Main Arguments

 &emsp;&emsp; 本篇论文提出了一种新的文本对抗攻击方法，是通过删除或替换文本中重要或突出的单词或在文本样本中引入新单词来实现的。该算法最适合在每个示例类中都有子类别的数据集。

## 4. Framework

 &emsp;&emsp; 作者采用了替换、插入、删除策略来进行修改，算法步骤如下：

![4](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/4.png?raw=true)

### 4.1 Calculate contribution of each word  

 &emsp;&emsp; 如果将一个词移除后，类被概率发生了很大的改变，则说明这个单词是很重要的。因此，我们这样来定义一个单词的重要程度：
$$
\mathcal{C}_{F}\left(w_{k}, y_{i}\right)=\left\{\begin{array}{l}
p_{F}\left(y_{i} \mid s\right)-p_{F}\left(y_{i} \mid s^{\mid w_{k}}\right), \text { if } F(s)=F\left(s^{\mid w_{k}}\right)=y_{i} \\
p_{F}\left(y_{i} \mid s\right)+p_{F}\left(y_{j} \mid s^{\mid w_{k}}\right), \text { if } F(s)=y_{i} \text { and } F\left(s^{\mid w_{k}}\right)=y_{j}
\end{array}\right.
$$
 &emsp;&emsp; 但是，对于一个大型的文本句子来说，上述的计算太耗时间，因此，作者又使用了FGSM的思想来估计单词的重要度：
$$
\mathcal{C}_{F}(w_k,y) = -\nabla_{w_k}J(F,s,y_i)
$$
 &emsp;&emsp; 其中，$\ y_i$ 表示$\ s$ 的真是标签，$\ J$ 表示损失函数。在实际用的时候，我们是可以直接得到梯度$\ -\nabla_{s}J(F,s,y_i)$ 的。所以，这种方法比较快。

### 4.2 Build candidate pool P for each word in sample text  

 &emsp;&emsp; 在有了这些分数之后，我们按照分数从大到小的顺序，依次的来修改单词。对于每个单词，我们需要为其建立候选池，我们可以考虑该词的同义词、拼写错误的词或者考虑类型或子类别特定的关键字。

### 4.2.1 Synonyms and typos  

 &emsp;&emsp; 使用同义词、以及容易拼写错误的单词，比如good单词容易拼写成god，将god加入good的候选池。

### 4.2.2 Genre specific keywords  

 &emsp;&emsp; 对于分类来说，比如情感分类，某些词可能对某一特定类型的电影产生积极的情绪，但对其他类型的电影可能会强调消极情绪。这些关键词通过考虑语料库中的词频（term frequencies, tf）来捕捉类的独特属性，如果一个词的次品对于一个特定类别的样本文本来说是高的，而对于属于不同类别的文本来说是低的，那么我们可以有把握地说，这个词对于第一类文本来说是独特的。我们用$\ \delta_i$ 来表示第$\ i$ 个类别中的独特的关键词。另外，我们考虑了单词的类型信息，将$\ \delta_{i,k}$ 表示为第$\ i$ 类别，第$\ k$ 类型的独特的关键词。我们扩充候选池：
$$
P = P\ \cup \{\delta_j\ \cap \delta_{i,k} \}
$$
 &emsp;&emsp; 其中，$\ i \ne j$ ，$\ k$ 表示该单词的类型index（喜剧、恐怖片等）。这里我们其实考虑的是IMDB中两个类别的公共部分，加入到候选集中。这样，我们就忽略了类别独特的部分（sub-category）信息。所以，当我们添加这些单词到句子的时候，他们会往中间移动。

### 4.3 Crafting the adversarial sample  

 &emsp;&emsp; 接下来，我们将描述，如何进行删除、添加和替换操作，假设原始样本为$\ s$ ，修改后的对抗样本为$\ s'$ ，目前正在操作的单词是$\ w_i$ 。我们按照每个单词的重要程度$\ C_F(w_i,y)$ 的顺序来处理单词。迭代的进行处理，直到生成对抗样本。

#### 4.3.1 Removal of words  

 &emsp;&emsp; 如果，$\ w_i$ 是副词，而且他的重要性$\ C_F(w_i,y)$ 很高，那么我们就移除这个单词。这是因为，副词起的是着重强调的意思，删除或者添加副词不会对句子的语法产生影响。

#### 4.3.2 Addition of words  

 &emsp;&emsp; 如果不满足4.3.1的条件，那么我们就从候选集中选择一个单词，我们选择：
$$
j = \arg\min_k\ C_F(p_k,y)
$$
 &emsp;&emsp; 如果，$\ p_j$ 是副词而$\ w_i$ 是形容词的话，那么我们就在$\ w_i$ 的前面插入$\ p_j$ 。

#### 4.3.3 Replacement of word  

 &emsp;&emsp; 如果不满足上面两个条件，那么，我们就是用$\ p_j$ 来替换$\ w_i$ 。但是，如果$\ p_j$ 是从Genre specific keywords  集合中得到的话，只有$\ p_j$ 与$\ w_i$ 词性相同的时候，我们才进行替换，如果不相同，那么我们就从候选集中找下一个合适的词来替换。词性相同用来保证我们生成的句子不容易被人类发现。

## 5.Result

### 5.1 Performance

![5](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/5.png?raw=true)



## 6. Argument

 &emsp;&emsp; 对单词的删除、替换和添加都做了条件，同时考虑了子类别信息，引入了公共关键词，使得生成的对抗样本更具备攻击性。

## 7. Further research





# 三、Deep Text Classification Can be Fooled

## 1. Paper Information

> 时间：2018年
>
> 关键词：Adversarial Attack，White-box，Black-box，Word-level，Character-level，Target Attack
>
> 论文位置：https://arxiv.org/ftp/arxiv/papers/1704/1704.08006.pdf
>
> 引用：Liang B, Li H, Su M, et al. Deep text classification can be fooled[J]. arXiv preprint arXiv:1704.08006, 2017.

## 2. Motivation

## 3. Main Arguments

 &emsp;&emsp; 本论文提出一种新的有效地生成文本对抗样本的方法，使用了插入、修改和删除的策略，在白盒攻击和黑盒攻击场景下获得了对抗样本。在白盒攻击下，我们利用梯度信息来决定如何、在哪进行插入、修改和删除。在黑盒攻击下，我们生成一些测试样本进行探测来获得上述的信息。

## 4. Framework

### 4.1 White-box Attack

 &emsp;&emsp; 在白盒攻击中，我们首先利用梯度信息，识别文本中哪个部分是重要的，利用这些信息来决定如何修改。

#### 4.1.1 Identifying Classification-important Items

 &emsp;&emsp; 对于chartacter-level来说，**我们利用梯度信息得到影响句子最大的字符，我们选择前50个为hot字符，包含三个以上hot字符的单词被称为hot单词，两个相邻的hot单词组成hot短语（hot单词也属于hot短语）**。对于word-level来说，我们可以直接利用梯度信息得到hot单词。我们按照出现的次数排序，**将最频繁出现的hot短语称之为Hot Training Phrases (HTPs)**。HTPs阐明了插入什么，但是没有说明在哪里插入、删除和修改。

 &emsp;&emsp; **我们又利用梯度信息来定位那些对当前分类最有贡献的短语，将其称为Hot Sample Phrases（HSPs）**。HSPs阐明了在哪里进行操作。

#### 4.1.2 Attacking Character-level DNN

 &emsp;&emsp; 主要使用了三种策略，插入、修改和删除，这里针对的是目标攻击。

##### 4.1.2.1 Insertion Strategy

 &emsp;&emsp; 在插入时，我们将HTP插入到HSP前，如：

![6](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/6.png?raw=true)

 &emsp;&emsp; 当然，我们也会遇到插入很多个HTP的情况，这会使对抗样本的可读性变差。为了解决这个问题并丰富攻击方法，我们**引入了自然语言处理的水印技术**，比如同义词或拼写错误单词的替换、释义表示（paraphrasing representation ）、增加预设（adding presuppositions ）、插入语义空短语（inserting semantically empty phrases）。事实上，对抗扰动也可以被视为一种水印技术，用类似的方式嵌入到文本中。

 &emsp;&emsp; 在这里，我们通过插入预设和语义空短语来扰动目标文本。预设是读者所熟知的隐含信息，一个语义空洞的短语是必不可少的组成部分，不管有没有它们，文本的意义都没有改变。通常，我们考虑通过将它们组装到一个语法单元中并将其插入到适当的位置来引入多个HTP。新单元可以被设计成可有可无的事实(见图3)，甚至可以被设计成不损害文本主要语义的伪造事实(见图4)。

![7](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/7.png?raw=true)

 &emsp;&emsp; 具体来说，通过搜索互联网或一些事实数据库，我们可以得到一些与插入点密切相关的事实，也可以包含目标类的一些理想的HTP。例如，如图3所示，我们谷歌搜索“YG Entertainment”，我们可以从维基百科中很容易得到公司类的三个HTP(“company”、“founded”和“Entertainment”)。在公司名称后插入它可以制造一个有效的对抗样本，而不会引起人类观察者的注意。一个适当的事实是不可得时，我们提出一个新的概念，称为伪造事实，包装理想的HTPs。伪造的事实可以通过改造一些与HTPs有关的真实事物来创造，使人们相信它是真实发生的。此外，我们排除了伪造事实，可以通过检索相反的证据在互联网上被推翻。图4显示了一个伪造的事实，它携带了令人满意的HTP(“romantic”, “movie”, “directed by” and “American”)，愚弄了目标DNN。

##### 4.1.2.2 Modification Strategy

 &emsp;&emsp; 我们使用同义词或者容易写出的错别拼写或者是一些相似的字符（比如小写的l与数字1）来替换HSP。

##### 4.1.2.3 Removal Strategy

 &emsp;&emsp; 删除策略往往不是那么有效，但会极大地降低原始类别的置信度。

#### 4.1.3 Attacking Word-level DNN

 &emsp;&emsp; Word-level的攻击与character-level很相似

### 4.2 Black-box Attack

 &emsp;&emsp; 在黑盒模型中，我们无法获得梯度信息。我们利用了fuzzing（模糊测试）技术来实现对HTP和HSP的定位。通过复杂地生成大量畸形输入，fuzzing可以触发意想不到的系统行为(例如，系统崩溃)，从而发现潜在的安全漏洞，甚至在不了解目标系统详细信息的情况下。同样，在我们提出的方法中，有目的地生成一些测试样本来探测目标模型。

 &emsp;&emsp; 给定样本为种子样本，我们用空格来mask掉某个单词送入样本，检查该样本与种子样本的分类结果差异，来判断重视程度，从而确定HTP和HSP。

 &emsp;&emsp; **不论是白盒攻击还是黑盒攻击，我们会记录每个类别的HTP，在攻击时，使用不同类别的HTP进行插入、删除或替换从而实现目标攻击。**

## 5.Result

![9](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97/%E6%96%87%E6%9C%AC%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB%E4%B9%8BWord-Level%E7%AC%94%E8%AE%B0%E7%B3%BB%E5%88%97%EF%BC%88%E4%B8%80%EF%BC%89/9.png?raw=true)

## 6. Argument

 &emsp;&emsp; 检测了不同类别的Hot词语，通过插入、删除或修改这些Hot词语进入样本实现目标攻击，但是实验样本数目太少了。

## 7. Further research

