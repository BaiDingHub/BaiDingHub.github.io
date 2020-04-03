---
title: Ubuntu安装NVIDA、CUDA、cuDNN
date: 2020-04-03 10:47:05
tags:
 - [Linux]
 - [软件安装]
categories: 
 - [教程,Linux]
keyword: "Linux,NVIDA,CUDA,cuDNN"
description: "Ubuntu安装NVIDA、CUDA、cuDNN"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Ubuntu%E5%AE%89%E8%A3%85NVIDA%E3%80%81CUDA%E3%80%81cuDNN/cover.jpg?raw=true
---

<meta name="referrer" content="no-referrer"/>

## 1.先安装好gcc，g++，make

```bash
sudo apt-get install gcc
```

```bash
sudo apt-get install g++
```

```bash
sudo apt-get install make
```

```bash
sudo apt-get update
```
## 2.安装NVIDA

 ##### 1）去[NVIDA官网](https://www.nvidia.com/Download/index.aspx?lang=en-us)查找并下载相应的显卡驱动.run文件
 #####   卸载原有驱动的方法（若未安装过驱动可跳过）

```bash
#for case1: original driver installed by apt-get:
sudo apt-get remove --purge nvidia*

#for case2: original driver installed by runfile:
sudo chmod +x *.run
sudo ./NVIDIA-Linux-x86_64-384.59.run --uninstall

```

 ##### 2）关闭nouveau


```bash
lsmod | grep nouveau
```
观察是否有输出
若有输出

```bash
sudo vim /etc/modprobe.d/blacklist.conf
```
在最后一行添加

```bash
blacklist nouveau
```
之后

```bash
sudo update-initramfs -u
```
重启

输入
```bash
lsmod | grep nouveau
```
若没有输出，则证明成功

##### 3)安装驱动
按下Ctrl+Alt+F2进入命令行界面

然后关掉图形界面，具体方法可查看[这篇博客](https://blog.csdn.net/StardustYu/article/details/85109013)

进入你下载的.run文件的目录

先赋予权限

```bash
sudo chmod a+x *.run
```
运行.run文件进行安装(注意参数)

```bash
sudo ./*.run –no-opengl-files
```
参数说明

`–no-opengl-files` 只安装驱动文件，不安装`OpenGL`文件。这个参数最重要
`–no-x-check` 安装驱动时不检查X服务
`–no-nouveau-check` 安装驱动时不检查`nouveau`
后面两个参数可不加。
安装过程都默认`yes`

安装完成后`reboot`重启

安装完成后，输入

```bash
nvidia-smi
```
若有输出，则证明安装成功

##### 4）一些注意事项

 - 在安装前，要关闭电脑的secure boot。进入bios，进入BOOT SETUP，进入Security'，将secure boot 设置为disabled。
 - 若在安装过程中出现gcc，make等词汇，说明gcc，make等未安装

## 3.安装CUDA
##### 进入[CUDA官网](https://developer.nvidia.com/cuda-toolkit-archive)下载相应版本的.run文件

###### 根据官网提示安装CUDA的.run文件

```bash
sudo sh cuda_*.run
```

   

在安装过程中除了安装驱动选项选择`no`，其他选择`yes`或默认
<br>
若安装过程出现tmp挂载盘容量不够，则可使用如下命令

```bash
sudo sh cuda_*.run --tmpdir=/home/ --override
```
10.1+版本在安装界面发生了很大的改变，在选择安装界面，应修改为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323204155313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
如果想要修改cuda的安装目录，则在Options内进行配置，记住，在修改了cuda的安装目录后，后面的环境配置的目录也要相应的修改。

##### 若出现missing recommended libraries错误
安装依赖

```bash
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```
之后再次安装即可

##### 配置环境变量

打开`.bashrc`文件  

```bash
sudo vim ~/.bashrc

如果用的zsh , 打开.zshrc文件
sudo vim ~/.zshrc
下面的文件作相应修改
```

在文件结尾加上

```bash
export PATH="/usr/local/cuda-10.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
```

其中`cuda-10.0`应改为相应版本

使`bashrc`文件生效

```bash
source ~/.bashrc
```

##### 验证

输入`nvcc -V`验证能否查看`CUDA`版本

卸载

在`/usr/local/cuda/bin`目录下运行`cuda`自带的卸载工具`uninstall_cuda_*.pl`

```bash
sudo ./uninstall_cuda_*.pl
```

## 4.安装cuDNN
进入[cuDNN下载官网](https://developer.nvidia.com/rdp/cudnn-download)下载相应版本的`cuDNN`

选择`cuDNN Library for Linux`下载

参考`cuDNN Installation Guide`进行安装

解压下载的`.tgz`文件

```bash
tar -xzvf cudnn-*.tgz
```

将解压出的文件拷贝到`CUDA`安装目录

```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include

sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## 5.cuda的卸载
卸载CUDA很简单，一条命令就可以了，主要执行的是CUDA自带的卸载脚本，读者要根据自己的cuda版本找到卸载脚本：

```bash
sudo /usr/local/cuda-8.0/bin/uninstall_cuda_8.0.pl
```

卸载之后，还有一些残留的文件夹，之前安装的是CUDA 8.0。可以一并删除：

```bash
sudo rm -rf /usr/local/cuda-8.0/
```


