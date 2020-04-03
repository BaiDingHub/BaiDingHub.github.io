---

title: Linux中anaconda3的下载与安装
date: 2020-04-03 10:46:05
tags:
 - [Linux]
 - [软件安装]
categories: 
 - [教程,Linux]
keyword: "Linux,anaconda3,软件安装"
description: "Linux中anaconda3的下载与安装"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Linux%E4%B8%ADanaconda3%E7%9A%84%E4%B8%8B%E8%BD%BD%E4%B8%8E%E5%AE%89%E8%A3%85/cover.jpg?raw=true
---



## Linux中anaconda3的下载与安装

 1. 去[anaconda官网](https://www.anaconda.com/download/#linux)下载anaconda3最新版本



  ![在这里插入图片描述](https://img-blog.csdn.net/20180926092103289?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

  2.在文件系统的下载中找到这个文件



  ![在这里插入图片描述](https://img-blog.csdn.net/20180926092256672?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



3.右键打开终端，输入`sh Anaconda*sh`（版本根据自己下载的anaconda进行更改）
4.根据提示输入回车



![在这里插入图片描述](https://img-blog.csdn.net/20180926093422133?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

5.查看注册信息，阅读完后输入yes



![在这里插入图片描述](https://img-blog.csdn.net/20180926093500126?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

6.回车进行安装（记住默认安装位置，以后可能用到）



![在这里插入图片描述](https://img-blog.csdn.net/20180926093605393?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

7.输入yes加入环境变量



![在这里插入图片描述](https://img-blog.csdn.net/20180926093647254?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

8.安装完成

##  检测是否安装成功

1.打开终端，输入python，是否出现anaconda



![在这里插入图片描述](https://img-blog.csdn.net/2018092609383097?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2.如果没有，在终端输入
source /etc/profile

3.如果依然不行，则
vim /etc/profile
在文末添加
export PATH="/opt/anaconda3/bin:$PATH"（PATH内为anaconda的默认安装位置）
然后输入
source /etc/profile

##  jupyter的使用
打开终端，输入jupyter notebook，若你使用的谷歌浏览器，输入完成后，弹出谷歌浏览器的默认窗口，而没有显示jupyter界面，建议将firefox设置为默认浏览器


