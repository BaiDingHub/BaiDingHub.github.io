---
title: markdown之Typora的下载
date: 2020-04-03 10:55:05
tags:
 - [markdown]
categories: 
 - [教程,其他软件]
keyword: "markdown,Typora"
description: "markdown之Typora的下载"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%85%B6%E4%BB%96%E8%BD%AF%E4%BB%B6/markdown%E4%B9%8BTypora%E7%9A%84%E4%B8%8B%E8%BD%BD/cover.png?raw=true
---



# markdown之Typora的下载

 ### 在[Typora官网](https://www.typora.io/)下载，并默认安装
![在这里插入图片描述](https://img-blog.csdn.net/20180925162541897?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 ###  在windows中设置右键快捷建立Typora文件
    1.新建txt文件，并在文件中写入如下内容
```
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\.md\ShellNew]

"NullFile"=""

"FileName"="template.md"
```

  2.将此txt文件另存为，并注意格式
  ![在这里插入图片描述](https://img-blog.csdn.net/20180925162757693?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

   3.文件将变成reg文件，双击运行
   ![在这里插入图片描述](https://img-blog.csdn.net/20180925162832115?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

  4.重启电脑


### 在linux中安装typora
1.
`sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA300B7755AFCFAE`

2.

```
sudo add-apt-repository 'deb http://typora.io linux/'
```
3.

```
sudo apt-get update
```
4.

```
sudo apt-get install typora
```

