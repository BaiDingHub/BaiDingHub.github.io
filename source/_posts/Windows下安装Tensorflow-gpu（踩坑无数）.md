---
title: Windows下安装Tensorflow-gpu（踩坑无数）
date: 2020-04-03 10:56:05
tags:
 - [Windows]
 - [Tensorflow]
categories: 
 - [教程,其他软件]
keyword: "Windows,Tensorflow"
description: "Windows下安装Tensorflow-gpu（踩坑无数）"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%85%B6%E4%BB%96%E8%BD%AF%E4%BB%B6/Windows%E4%B8%8B%E5%AE%89%E8%A3%85Tensorflow-gpu%EF%BC%88%E8%B8%A9%E5%9D%91%E6%97%A0%E6%95%B0%EF%BC%89/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

当开始学习深度学习的时候，自然就要用到tensorflow-gpu
版，而安装是个巨坑。博主曾深深陷入其中无法自拔，最终破釜沉舟，终于成功，哈哈哈哈哈。
以下是在windows中安装tensorflow的gpu版本的教程
# windows下安装tensorflow -gpu
## 1.安装cuda
首先要去[cuda官网](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)下载cuda
![在这里插入图片描述](https://img-blog.csdn.net/2018100623024590?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**强烈推荐默认安装地址**
安装步骤如下

![在这里插入图片描述](https://img-blog.csdn.net/20181006230822553?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/20181006230832768?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![在这里插入图片描述](https://img-blog.csdn.net/20181006230840256?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 2.安装cudnn
安装完cuda后，就去[cudnn官网](https://developer.nvidia.com/rdp/cudnn-download)下载cudnn。博主在当时下载的9.2版本。
![在这里插入图片描述](https://img-blog.csdn.net/20181006231053801?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

下载完成后，会得到一个压缩包，把压缩包里的文件copy到之前安装cuda的位置（如果默认安装的话，位置应该是C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0）
要对应着来copy。
将.h .lib 和.dll 文件分别拷贝到cuda的include, lib/x64, bin 文件夹下

## 3.安装完这两项之后，就是环境的配置

计算机上点右键，打开属性->高级系统设置->环境变量，可以看到系统中多了CUDA_PATH和CUDA_PATH_V8_0两个环境变量，接下来，还要在系统中添加以下几个环境变量：  


    CUDA_SDK_PATH = C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0(这是默认安装位置的路径) 
    CUDA_LIB_PATH = %CUDA_PATH%\lib\x64  
    CUDA_BIN_PATH = %CUDA_PATH%\bin  
    CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64  
    CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64

然后
在系统变量 PATH 的末尾添加：  


    %CUDA_LIB_PATH%;%CUDA_BIN_PATH%;%CUDA_SDK_LIB_PATH%;%CUDA_SDK_BIN_PATH%;  
      再添加如下4条（默认安装路径）： 
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64； 
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin； 
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\lib\x64； 
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\bin\win64； 

如图：
![在这里插入图片描述](https://img-blog.csdn.net/20181006231617216?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![在这里插入图片描述](https://img-blog.csdn.net/2018100623160879?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 4.安装tensorflow
直接在cmd中输入

```
pip install tensorflow-gpu
```
如果嫌弃直接安装太慢，可以换个源

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu
```
安装成功后，在cmd进入python环境（博主用的anaconda环境）
输入

```
import tensorflow as tf
```
如果没有任何bug出现，那么恭喜你，你太幸运了。
然后你可以输入

```
tf.Session()
```
如果出现
![在这里插入图片描述](https://img-blog.csdn.net/20181006232540185?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
恭喜你tensorflow-gpu安装成功了。
当然，如果你没有那么顺利，也超级正常。
那就看之后的内容。
##  5.安装过程中遇到的一系列的坑
**首先**，最简单的，
如果你输入tf.Session()后没有出现，gpu的一系列信息。

```
pip show tensorflow-gpu
```
查看有没有tensorflow-gpu的信息
也可以输入

```
pip show tensorflow
```
查看是不是系统里面有tensorflow的cpu版本

如果有cpu版本，那么就卸载了tensorflow的cpu版本，重新安装一个。

```
pip uninstall tensorflow
```

**其次**，
如果你在import tensorflow时出现了import numpy出错，说明你的numpy版本不适合你的tensorflow版本。
那么，你就要更新你的numpy版本

```
pip install numpy --upgrade
```
之后发现，这个错误就没了。
当然，可能会是一些其他库的警告，比如
![在这里插入图片描述](https://img-blog.csdn.net/2018100710034215?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
仔细看一下这个错误，看一下哪个库出现问题了，这种情况一般就是更新一下相应的库就OK，比如这个错误就是h5py版本太低，所以我更新了h5py

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple h5py --upgrade
```

**最后**，
如果你import tensorflow时出现下面的信息
![在这里插入图片描述](https://img-blog.csdn.net/20181006233333372?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
那么，别慌，你肯定去到处去网上搜索资料，这个东西超级难查，博主曾被折磨的醉仙欲死。
解决这个问题，其实so easy。
就用最暴力的方法。
那就是进入E:\ComputerScience\software\Anaconda3\Lib\site-packages这个目录下，把所有有关tensorflow的东西全删了（就是这么暴力）。
然后重新安装tensorflow-gpu

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu
```
搞定！！！，哈哈哈哈
不知道你搞定了没有，反正我搞定了，哈哈哈。
欢迎没有搞定的人在下面讨论。。


## Linux 安装 tensorflow-gpu

预先安装好NVIDIA

    pip install --index-url http://mirrors.aliyun.com/pypi/simple/ tensorflow-gpu

