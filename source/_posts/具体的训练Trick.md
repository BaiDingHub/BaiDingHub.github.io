---
title: 具体的训练小技巧
date: 2020-04-03 11:10:05
tags:
 - [AI小知识]
 - [训练技巧]
categories: 
 - [深度学习,AI小知识]
keyword: "深度学习,AI小知识,"
description: "具体的训练小技巧"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/AI%E5%B0%8F%E7%9F%A5%E8%AF%86/%E5%85%B7%E4%BD%93%E7%9A%84%E8%AE%AD%E7%BB%83Trick/cover.png?raw=true
---





# 1、参数初始化

几种方式,结果差不多。但是一定要做。否则可能会减慢收敛速度，影响收敛结果，甚至造成Nan等一系列问题。 

优秀的初始化应该使得各层的激活值和状态梯度的方差在传播过程中的方差保持一致。不然更新后的激活值方差发生改变，造成数据的不稳定。



  **Xavier初始化** :

- 条件：正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。
- 论文：[http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf](https://link.zhihu.com/?target=http%3A//jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) 
- 理论方法：

$$
W \sim U[-\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}},\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}]
$$



- 假设激活函数关于0对称，且主要针对于全连接神经网络。**适用于tanh和sigmoid**。

**He初始化**：

- 条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变。

- 论文：[https://arxiv.org/abs/1502.01852](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1502.01852)

- 理论方法

  - 适用于ReLU的初始化方法：

  $$
  W \sim U[0,\sqrt{\frac{2}{\hat{n}_i}}]
  $$

  - 适用于Leaky ReLU的初始化方法：

  $$
  W \sim U[0,\sqrt{\frac{2}{(1+\alpha^2)\hat{n}_i}}]
  $$

  其中 
  $$
  \hat{n}_i = h_i * w_i * d_i  \\
  h_i,w_i分别表示卷积层中卷积核的高和宽 \\
  d_i表示当前层卷积核的个数
  $$









**具体方法**

下面的n_in为网络的输入大小，n_out为网络的输出大小，n为n_in或(n_in+n_out)/2

- uniform均匀分布初始化：

  ```
  w = np.random.uniform(low=-scale, high=scale, size=[n_in,n_out])
  ```

  - Xavier初始法，适用于普通激活函数(tanh,sigmoid)：`scale = np.sqrt(3/n)`
  - He初始化，适用于ReLU：`scale = np.sqrt(6/n)`

- normal高斯分布初始化：

  ```
  w = np.random.randn(n_in,n_out) * stdev # stdev为高斯分布的标准差，均值设为0
  ```

  - Xavier初始法，适用于普通激活函数 (tanh,sigmoid)：`stdev = np.sqrt(n)`
  - He初始化，适用于ReLU：`stdev = np.sqrt(2/n)`

- svd初始化：对RNN有比较好的效果。参考论文：[https://arxiv.org/abs/1312.6120](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6120)



**技巧**

正确初始化最后一层的权重。如果回归一些平均值为50的值，则将最终偏差初始化为50。如果有一个比例为1:10的不平衡数据集，请设置对数的偏差，使网络预测概率在初始化时为0.1。正确设置这些可以加速模型的收敛。



# 2、数据预处理方式

- **zero-center** 

  这个挺常用的.

  ```
  X -= np.mean(X, axis = 0)   # zero-center
  X /= np.std(X, axis = 0)    # normalize
  ```

- **PCA whitening** 

  这个用的比较少.



# 3、梯度裁剪

实现方法见**[pytorch小操作](https://blog.csdn.net/StardustYu/article/details/102856387)**





# 数据增强方法

该博客中的图片和代码来自其他博客，本博客做总结用

## 1.基本的数据增强方法

### 1）翻转Flip

 包括水平翻转和垂直翻转

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115173731529.png)

### 2）旋转Rotation

将图像旋转一个角度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115173743479.jpg)

### 3）平移Translations

上下左右移动物体
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115173748456.png)

### 4）随即裁剪crop

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011517374647.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)





### 5）加噪声--高斯噪声等

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115173758817.png)

### 6）放射变换

<br>

### 7）平滑模糊图像

<br>

### 8）颜色空间变换

<br>

### 9）随机擦除法（随机去掉一部分区域）

<br>

## 2.高阶方法

### 1）GAN自动生成

可用DCGAN，DCGAN效果更好一些
<img src=https://img-blog.csdnimg.cn/20200115173804496.png width="60%"/>

### 2）条件GAN 

通过冬天的图片生成夏天的图片

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115174033722.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

### 3）图片风格转移

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200115174040267.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

参考博客

<https://segmentfault.com/a/1190000016526917>

<https://zhuanlan.zhihu.com/p/41679153>