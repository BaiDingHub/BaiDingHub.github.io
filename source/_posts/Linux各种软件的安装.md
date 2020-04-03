---
title: Linux各种软件的安装
date: 2020-04-03 10:43:05
tags:
 - [Linux]
 - [软件安装]
categories: 
 - [教程,Linux]
keyword: "Linux,软件安装"
description: "Linux下安装各种软件"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Linux%E5%90%84%E7%A7%8D%E8%BD%AF%E4%BB%B6%E7%9A%84%E5%AE%89%E8%A3%85/cover.png?raw=true
---



**以Ubuntu18.04为例**

##  1.google浏览器的安装

参见[我的博客](https://blog.csdn.net/StardustYu/article/details/82845656)

## 2.anaconda的安装
参见[我的博客](https://blog.csdn.net/StardustYu/article/details/82848922)

## 3.markdown文档的安装
参见[我的博客](https://blog.csdn.net/StardustYu/article/details/82841260)

## 4.NVIDA，CUDA等的安装
参见[我的博客](https://blog.csdn.net/StardustYu/article/details/87883622)

## 5.WPS的安装
进入[wps官网](http://www.wps.cn/product/wpslinux)，下载linux版(.deb格式)，双击直接安装即可

## 6.网易云音乐的安装
进入网易云音乐官网，下载linux版(.deb格式)，双击直接安装即可

## 7.搜狗输入法的安装
进入[搜狗输入法官网](https://pinyin.sogou.com/linux/?r=pinyin)，下载linux版(.deb格式)，双击直接安装。

进入到系统设置，语言设置，进入Manage installed languages，即这个界面
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190311103911371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
将Keyboard input method system 更改为fcitx。

重启计算机，点击右上角的图标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190311104007711.png)
进入congfigure，点击左下角加号
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190311104120538.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
将Only Show Current Language取消勾选，搜索Sogou，添加即可。

## 8.pycharm的安装
**下载**

进入[pycharm官网](https://www.jetbrains.com/pycharm/download/#section=linux)下载

**安装**

打开刚才下载的目录。右击文件，点击提取到此处。

解压完成后，打开刚才解压好的文件夹，然后再打开bin目录。

在文件夹空白处右击，在此处打开终端然后输入：`sh ./pycharm.sh` 回车

接着就打开了pycharm。


在pycharm中的顶部菜单栏`tools -> Create desktop entry`。即可将pycharm加入菜单中

## 9.guake的安装

    sudo apt-get install guake

## 10.QQ的安装
下载[Wine-QQ](http://yun.tzmm.com.cn/index.php/s/XRbfi6aOIjv5gwj)

进入下载的文件夹，打开终端

    chmod a+x *.AppImage
    ./*.AppImage

或者

[https://github.com/wszqkzqk/deepin-wine-ubuntu](https://github.com/wszqkzqk/deepin-wine-ubuntu)

根据步骤安装deepin

然后下载QQ，双击安装即可

如果是64位系统，可能需要依赖支持

```
sudo dpkg --add-architecture i386
sudo apt-get update
# 可能需要添加下列32位库
sudo apt-get install lib32z1 lib32ncurses5
```

## 11.Shell软件（ZSH的安装）
**zsh安装**

```
sudo apt-get install zsh
```
**oh-my-zsh安装**

```
curl 方式:    
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" 

wget 方式:
sh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)" 
```
**切换默认Shell**

    chsh -s /bin/zsh

[主题配置方法](https://www.jianshu.com/p/0f3dcec21a97)

**安装autojump自动跳转插件**

```
sudo apt-get install autojump
vim .zshrc
#在最后一行加入，注意点后面是一个空格
. /usr/share/autojump/autojump.sh
source ~/.zshrc
```
**安装zsh-syntax-highlighting语法高亮插件**

```
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
echo "source ${(q-)PWD}/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc
source ~/.zshrc
```
**安装zsh-autosuggestions语法历史记录插件**

```
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
plugins=(zsh-autosuggestions)
vim ~/.zshrc
然后增加zsh的执行文件在最后一行：
source $ZSH_CUSTOM/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
source ~/.zshrc
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314173524640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

**zsh主题修改**

```
vim ~/.zshrc

修改ZSH_THEME为自己想要的主题
推荐avit

cd ~/.oh-my-zsh/themes
ls
来查看所有的可用主题，将上面的ZSH_THEME更改成相应的名字即可
可以修改 里面的文件，来对相应的主题进行修改
```

## 12.安装sublime

```
#安装GPG
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -

#确保apt被设置为https源
sudo apt-get install apt-transport-https

#选择稳定版本
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list

#安装sublime-text
sudo apt-get update
sudo apt-get install sublime-text
```
## 13.华科有线校园网客户端
(deb包)下载地址：

`MentoHUST V0.3.4 for Ubuntu i386`  与 `MentoHUST V0.3.4 for Ubuntu amd64` 下载

免费下载地址在 http://linux.linuxidc.com/

具体下载目录在 /2013年资料/1月/20日/Ubuntu下使用MentoHUST代替锐捷认证上网

解压后，运行`.deb`文件即可

```
sudo mentohust
```

配置如下

```
eric@eric-Satellite-C850:~/Downloads/mentohust_0.3.4-1_amd64$ sudo mentohust
欢迎使用MentoHUST	版本: 0.3.4
Copyright (C) 2009-2010 HustMoon Studio
人到华中大，有甜亦有辣。明德厚学地，求是创新家。
Bug report to http://code.google.com/p/mentohust/issues/list
 
** 网卡[1]:	eth0
** 网卡[2]:	wlan0
** 网卡[5]:	nflog
** 网卡[6]:	nfqueue
** 网卡[7]:	usbmon1
** 网卡[8]:	usbmon2
** 网卡[9]:	usbmon3
** 网卡[10]:	usbmon4
?? 请选择网卡[1-10]: 1
** 您选择了第[1]块网卡。
?? 请输入用户名: yourusername
?? 请输入密码: yourpassword
?? 请选择组播地址(0标准 1锐捷私有 2赛尔): 0
?? 请选择DHCP方式(0不使用 1二次认证 2认证后 3认证前): 2
** 用户名:	M201672859
** 网卡: 	eth0
** 认证超时:	8秒
** 心跳间隔:	30秒
** 失败等待:	15秒
** 允许失败:	8次
** 组播地址:	标准
** DHCP方式:	认证后
** 通知超时:	5秒
** DHCP脚本:	dhclient
!! 在网卡eth0上获取IP失败!
!! 在网卡eth0上获取子网掩码失败!
** 本机MAC:	00:26:6c:11:36:00
** 使用IP:	0.0.0.0
** 子网掩码:	255.255.255.255
** 认证参数已成功保存到/etc/mentohust.conf.
>> 寻找服务器...
** 认证MAC:	00:1a:a9:17:ff:ff
>> 发送用户名...
>> 发送密码...
>> 认证成功!
$$ 系统提示:	1.关于防范ONION勒索软件病毒攻击的紧急通知http://ncc.hust.edu.cn/tz12/945.jhtml
2.关于2017年暑假校园网对外服务的通知http://ncc.hust.edu.cn/tz06/948.jhtml
 
 
 
!! 打开libnotify失败，请检查是否已安装该库文件。
>> 正在获取IP...
>> 操作结束。
** 本机MAC:	00:26:6c:11:36:00
** 使用IP:	115.156.162.119
** 子网掩码:	255.255.254.0
>> 发送心跳包以保持在线...
```

## 14.音乐软件cocomusic的下载

Linux版音乐CoCoMusic最新版本是2.0.2，提供deb、tar.xz、AppImage等安装包，它被称为Linux版的QQ音乐，主要是因为它使用了QQ音乐的曲库，当然这款软件不会是自家的软件产品，只是一个热心的程序员无私的奉献精品。下面将为你带来CoCoMusic的安装方法及使用报告。

 

下载地址

[Cocomusic的下载地址](https://github.com/xtuJSer/CoCoMusic/releases)


<br>

## 15.vscode的下载
官网[https://code.visualstudio.com/](https://code.visualstudio.com/)

<br>


## 16.pip 更换下载源
以安装torch为例

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvison
```

