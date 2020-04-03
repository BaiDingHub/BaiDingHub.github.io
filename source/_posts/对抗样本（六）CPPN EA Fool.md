---
title: 对抗样本（六）CPPN EA Fool
date: 2020-04-03 13:06:05
tags:
 - [深度学习]
 - [对抗攻击]
categories: 
 - [深度学习,对抗样本]
keyword: "深度学习,对抗样本,CPPN EA Fool"
description: "对抗样本（六）CPPN EA Fool"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%EF%BC%88%E5%85%AD%EF%BC%89CPPN%20EA%20Fool/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 一、论文相关信息

## &emsp;&emsp;1.论文题目

&emsp;&emsp;&emsp;&emsp; **Deep neural networks are easily fooled: High confidence predictions for unrecognizable images**

## &emsp;&emsp;2.论文时间

&emsp;&emsp;&emsp;&emsp;**2015年**

## &emsp;&emsp;3.论文文献

&emsp;&emsp;&emsp;&emsp;[https://arxiv.org/abs/1412.1897](https://arxiv.org/abs/1412.1897)



# 二、论文背景及简介

 &emsp; &emsp; 之前的攻击方法都是假反例攻击，即都是将数据集的图片添加扰动而是分类器识别错误。这篇论文从**假正例攻击方式**出发，试图使用**进化算法或者梯度上升**来生成一个人类识别不出，但是却能让机器深信不疑的图片。这篇论文**阐明了人类视觉与目前的DNN的差别**，并提出了对DNN用于计算机视觉的泛化性的问题。

<br>

# 三、论文内容总结

- 首次提出**假反例攻击**，即：生成人类无法识别的图片，却能够让神经网络以高置信度分类成某个类别
- 使用了多种不同的方法生成图片，称为“fooling images”
  - 普通的EA算法，对图片的某个像素进行变异、进化
  - **CPPN EA算法**，可以为图像提供一些几何特性，如对称等
  - 梯度上升
- 将fooling images添加到数据集中，作为n+1类，重新训练网络，然后将得到的网络再生成fooling images，再训练，如此迭代，来提高模型的防御能力。
- 介绍了在MNIST数据集和ImageNet数据集上进行实验的区别
  - 在MNIST数据集上，由于数据量小，得到的网络模型的容量小，更容易生成fooling images，也跟难通过 利用fooling images重新训练模型的方式 来提高其防御能力
  - 在ImageNet数据集上，数据量大，类别多，得到的网络模型的容量大，更难生成fooling images，不过由于其容量大，能够通过重新训练的方式提高防御能力
- **判别式模型**$\ p(y|X)$ 与**生成式模型**$\ p(y,X)$ 的对比：判别式模型比生成式模型拥有更好的防御能力
- 假反例攻击为神经网络的部署提出了很大的应用问题。

附：如需继续学习对抗样本其他内容，请查阅[对抗样本学习目录](https://blog.csdn.net/StardustYu/article/details/104410055)

<br>

# 四、论文主要内容

## 1、介绍

 &emsp; &emsp;  近来，很多深度模型结构令人赞叹，在的出很好的结果的同时，也引出了一个问题，人类视觉系统和这些模型之间有什么区别。

 &emsp; &emsp;  L-BFGS那篇文章给出了一个解释，用人眼无法分辨的扰动来改变一张图片，却可以让DNN分类错误。

 &emsp; &emsp;  这篇文章提出了另一种来说明区别的方法，即：**生成一张人类无法分辨的图片，却能够让DNN以99%的置信度相信这是某个物体** 。我们使用**进化算法**或者**梯度上升**来得到这样的图片。同时，作者也发现，对于使用MNIST训练集训练的DNN来说，使用对抗样本进行finetune也不能让DNN拥有良好的防御能力，即使重新训练的DNN分类出一部分对抗样本后，我们仍然可以重新生成一批这个重新训练的DNN无法防御的对抗样本。

 &emsp; &emsp;  我们的发现，阐明了在人类视觉系统与计算机视觉系统的不同，同时引出了DNN的泛化性的问题。

<br>

## 2、方法

### 2.1 该论文所使用的模型

 &emsp; &emsp;  作者认为，小的架构和优化方式的不同不会影响我们的结果，因此，作者选用了**AlexNet**作为实验模型

 &emsp; &emsp;  为了说明，我们的结果在其他的深度模型和数据上的结果，我们也使用了在MNIST上训练的**LeNet**

<be>

### 2.2 用进化算法生成图片

 &emsp; &emsp;  进化算法（evolutionary algorithms）是达尔文进化论启发下的优化算法。它们包含一个“有机体”（这里是图像）群体，这些有机体交替进行面部选择（保持最佳状态），然后进行随机排列（变异和/或交叉）。选择哪种有机体取决于拟合函数，在实验中，拟合函数是DNN对属于某一类的图像所作的最高预测值。如图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174041851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp;  传统的进化算法只能够针对一个目标或者是一小部分目标，比如，优化图像去匹配ImageNet的一个类。在这里，我们使用了一个新的算法，叫做：**multi-dimensional archive of phenotypic elites MAP-Elites** ，这个算法可以让我们同时的进化出一个群体，如：ImageNet的1000个类。

 &emsp; &emsp;  **MAP-Elites的工作方式**如下：它为每一个类别保存一个最好的有机体（图像）。在每次迭代过程中，该算法会从群体（数据集）中随机的选择一个有机体，随机的变异它，如果他在某一类上的拟合程度比之前保存的更好，那么就替换掉之前的有机体。在这里，拟合函数是通过将图像传给DNN来得到的，即，如果图像在某一类上有更高的置信度，那么新生成的图像，就会替代之前在该类上的最好的那一个。

 &emsp; &emsp;  我们使用两种不同的编码来测试进化算法，来弄明白图片是如何跟基因组一样来进行操作的。

 &emsp; &emsp;  第一种是**直接编码** ，对于MNIST，28×28像素的每一个像素有一个灰度整数，对于ImageNet，256×256像素的每一个像素有三个整数（H，S，V）。每个像素值在[0，255]范围内用均匀随机噪声初始化。这些数字是独立变异的；首先通过以0.1开始（每个数字有10%的机会被选择变异）并以每1000次变异下降一半的速率来确定哪些数字是变异的。然后通过多项式变异算子对选择要变异的数字进行变异，变异强度固定为15。

 &emsp; &emsp;  第二种是**间接编码**，它更可能生成规则图像（包含可压缩模式的图像（例如，对称和重复））。间接编码的图像往往是规则的，因为基因组中的元素可以影响图像的多个部分。具体地说，这里的间接编码是一个**合成模式生成网络（CPPN）**，它可以演化复杂的、规则的图像，重新组合自然和人造物体。

 &emsp; &emsp;  由CPPN进化得到的图像可以被DNN识别，甚至可以被人类识别。如：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174058931.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp;  这些图片是在PicBreeder.org网站上制作的，在这个网站上，用户通过选择自己喜欢的图片作为进化算法中的适应度函数，这些图片成为下一代的父母。

 &emsp; &emsp;  CPPN类似于人工神经网络（artificial neural networks (ANNs)）。CPPN接收像素的（x，y）位置作为输入，并输出该像素的灰度值（MNIST）或HSV颜色值元组（ImageNet）。与神经网络一样，CPPN计算的功能取决于CPPN中的神经元数量、它们之间的连接方式以及神经元之间的权重。每个CPPN节点可以是一组激活函数中的一个（这里是正弦、sigmoid、高斯和线性），这些激活函数可以为图像提供几何规则性。例如，将x输入传递给高斯函数将提供左右对称性，将y输入传递给正弦函数将提供上下重复。进化决定了种群中每个CPPN网络的拓扑结构、权值和激活函数。

 &emsp; &emsp;  CPPN网络在刚开始的时候，是没有隐藏节点的，隐藏节点会随着时间来加进来，这让进化算法可以在变得复杂之前，能够找到简单的规则的图像。

 &emsp; &emsp;  实验是在Sferes进化计算框架上完成的。

<br>

## 3、结果

### 3.1  进化出不规则的图片来匹配MNIST

注意：这里的规则与不规则是上一节所指的直接与间接的编码，由于CPPN会为图像提供几何的规则性（对称等），所以称之为规则的，而直接编码没有几何性质，所以称为不规则的。还有一个区别是，CPPN可以进化出能够识别的图片，如上图。而直接编码进化出来的图片，无法识别，如下图。

 &emsp; &emsp;  我们直接对编码图片进行进化，来让LeNet将其分类为0~9之间(MNIST)。多次独立的进化过程反复产生的图像，让DNN能够有99%的置信度，但让人类无法识别。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174129965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
 &emsp; &emsp;  在50轮迭代后，就有图片能够置信度达到99.99%，经过200轮的迭代后，DNN的置信度已经平均有99.99%

<br>

### 3.2 进化出规则的图片来匹配MNIST

 &emsp; &emsp;  因为CPPN编码能够生成让人能够识别的图片，所以我们来测试，是否这种规则会给我们带来更好的结果。其生成的图片如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174147309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp;  经过一小轮的迭代，有图片能够置信度达到99.99%，经过200轮的迭代后，DNN的置信度已经平均有99.99%。

 &emsp; &emsp;  由图片我们可以看到，被分类为1的图片，会有竖直的线，被分类为2的图片，在底部会有水平的线。结果表明，EA利用了DNN所学习的特征来生成的图片。

<br>

### 3.3 进化出不规则的图片来匹配ImageNet

 &emsp; &emsp;  我们对大数据集进行了测试。发现，结果并不成功，哪怕20000次迭代后，也没有为很多类生成高置信度的图片，平均置信度21.59%。当然也有一些成功的，进化算法为45个类生成了置信度超过99%的图片。在各类上的置信度如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174207621.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

<br>

### 3.4 进化出规则的图片来匹配ImageNet

 &emsp; &emsp;  这次使用CPPN来进化图片。

 &emsp; &emsp;  在五次独立的迭代后，CPPN生成了许多置信度大于99.99%的图片，但是无法被人识别。在经过5000轮迭代后，平均置信度达到了88.11%。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032817423251.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp;  尽管，人类无法识别由CPPN生成的图片，但是图片中确实包含一些类别的特征。也就是说，我们只需要去生成一个类中的需要的特征即可，而不需要将所有的特征都生成出来。

 &emsp; &emsp;  我们仅仅只需要有这些关键的**局部特征** 就行，这就让我们产生的**图片具有惊人的多样性**。这个多样性是非常显著的，这是因为：1、对图像的微小扰动可以改变DNN所判别的类标签，因此可能是因为**进化为所有的类别产生了非常相似的、高置信度的图像**。2、**许多图片在系统学上是相互关联的，这也就导致进化产生了相似的图像**，如图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174244283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
 &emsp; &emsp; 这张图，是指，一种图像可能同时对三个类别产生较高的置信度，但是我们通过进化，会让图像向不同的方向进化，从而产生不同的图像类型。这说明，我们使用的是每个类别的不同的特征。

 &emsp; &emsp; 为了**测试重复是否提高了DNN给出的图像的置信度，或者重复是否仅仅源于CPPN倾向于生成规则图像这一事实**。我们去除了一些重复的元素来看是都DNN的置信度是否下降。在大多数情况下，置信度确实下降了，这说明，重复可以让DNN的置信度提高。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174258423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 这个结果表明，**DNN倾向于去学习低层和中层特征，而不是学习对象的全局结构**。DNN能够正确地学习全局结构，那么如果图像中包含了自然图像中很少出现的对象子组件的重复，则图像的DNN置信度得分会更低。

 &emsp; &emsp; 在这些生成的图片中，对猫和狗的攻击的表现比较差，作者说明了两个可能的解释：1、数据集中的猫和狗的图片更多一些，也就意味着，它过拟合程度更低，也意味着其更难去欺骗，这个方面说明，数据集越大，网络越难被攻击。2、另一个可能的解释是因为ImageNet有太多猫和狗的类了，EA对每一个类很难找到最有代表性的图片（在该类置信度高，在别的类置信度低），而这对于以softmax为分类器的DNN来说是很重要的，这个方面说明，数据集的类别越多，越难被攻击。

<br>

### 3.5 进化得到的图片具有泛化性

 &emsp; &emsp; 之前，我们用EA得到了DNN对某一类的特征，那么是否所有的DNN对这一类别都是这种特征呢？我们可以通过对其他DNN进行测试来证明这一点。我们进行了两个实验，我们用DNN A来生成图片，然后输入到DNN B中，第一种情况是，DNN A和DNN B具有相同的结构和训练方法，但是初始化方式不同。第二种情况是，两者用相同的数据集训练，但具有不同的结构。在MNIST和ImageNet上都做了这个实验。

 &emsp; &emsp; 结果显示，图片在DNN A 和DNN B 上的置信度都大于99.99%，因此**DNN的泛化性可以被EA利用**。

<br>

### 3.6 训练网络来识别这些图片（防御方法）

 &emsp; &emsp; DNN可以很容易的被我们训练好的图片愚弄，为了进行对抗防御，那么我们应该**重新训练一下网络，使得网络将对抗样本分类为"fooling images"类别，而不是之前的任何类**。

 &emsp; &emsp; 作者在MNIST和ImageNet上对上述假设进行测试，操作如下：我们现在某个数据集上训练一个DNN 1，使用CPPN得到进化的图片，然后我们把这些图片添加到数据集上，标签为第n+1类。然后我们在新的数据集上训练DNN 2。

 &emsp; &emsp; 之后又对上述过程进行了优化，又在DNN 2的基础上，使用CPPN得到图片，将得到的图片仍然记为n+1类，然后不断地迭代。在每次迭代过程中，只添加n+1个类当中的m类（随机选取）的优化的图片。

<br>

### 3.7 在MNIST数据集上进行防御

 &emsp; &emsp; MNIST数据集总共60000张图片，共10个类别，平均每个类别6000张。

 &emsp; &emsp; 我们在第一次迭代中添加了6000张进化得到的图片（经过了3000次迭代）。每次新的迭代，我们都会增加1000张新得到的图片到训练集中。

 &emsp; &emsp; 作者发现，在重复操作后，LeNet的防御能力仍然很一般，我们仍然可以通过CPPN生成置信度99.99%的图片。

<br>

### 3.8 在ImageNet数据集上进行防御

 &emsp; &emsp; 这里是使用的ISLVRC（ImageNet的子数据集），共1281167张图片，1000个类别，平均每个类别有1300张。

 &emsp; &emsp; 我们在第一次迭代中添加了9000张进化得到的图片。每次新的迭代，我们都会增加7*1300张新得到的图片到训练集中。如果没有这种数据的不平衡，那么使用反例来训练网络是没有用处的。

 &emsp; &emsp; 在ImageNet上，DNN 2比DNN 1更难通过CPPN生成图片，且置信度下降。**ImageNet的网络更容易去区分CPPN生成的图片与原始数据集，因此更容易通过反例的训练来提高自己的防御能力**。

<br>

### 3.9 通过梯度上升来生成图片

 &emsp; &emsp; 另一种去生成图片的方式，是在**像素空间内的梯度上升** 。

 &emsp; &emsp; 我们会计算DNN的softmax的输出对于输入的梯度，我们根据梯度来增加一个单元的激活值，目的是去找到一个图片能够得到很高的分类置信度。

 &emsp; &emsp; 这种方法相比于之前介绍的两种EA算法，能够生成更多不同的样本。

<br>

## 4、讨论

 &emsp; &emsp; 通过进化算法，按道理来说，应该会出现两种情况，一种是我们能够为每一个类进化出人类可识别的图像。第二种是，考虑到局部最优的情况，生成的图片应该会在所有的类上的置信度都比较低。

 &emsp; &emsp; 但事实却与上述相反，我们得到的图片DNN可识别而人类无法识别，且置信度在某些类别上很高，而在某些类别上很低。作者认为这是**判别模型和生成式模型的不同导致的**。

 &emsp; &emsp; 判别式模型，即$\ p(y|X)$ ，给X，来求y。在这种模型的高维输入空间中，判别模型分配给类的区域可以远大于该类的训练样本所占用的区域（参见下图）。远离决策边界和深入分类区域的合成图像可能产生高置信度的预测，即使它们远离类中的自然图像。相关研究证实并进一步研究了这一观点，该研究表明，由于某些判别模型的局部线性性质和高维输入空间的结合，它们具有很大的高置信区间。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200328174324867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 &emsp; &emsp; 生成式模型，即$\ p(y,X)$ ，不仅要计算$\ p(y|X)$ 而且还要计算$\ p(X)$ 。这样的模型更难被攻击。因为，当出现对抗样本时，他们的$\ p(X)$ 会很低。当$\ p(X)$ 较低时，DNN在此类图像的标签预测中的置信度会大打折扣。不幸的是，目前的通用模型并不能很好地扩展到像ImageNet这样的数据集的高维性，所以测试它们在多大程度上被愚弄必须等待生成模型的发展。

 &emsp; &emsp; CPPN EA可以看作是一种能够可视化DNN所学习到的特征的一种新奇的方法。

<br>

## 5、总结

 &emsp; &emsp; 我们证明了判别式DNN模型更容易被欺骗。两种不同的进化算法能够生成大量的不同类型的人类无法识别的"fooling images"，梯度上升也能够生成。这些fooling images揭示了人类视觉系统与DNN的识别之间存在的差异。提出了dnn的泛化能力和使用dnn的解决方案的潜在成本的问题
