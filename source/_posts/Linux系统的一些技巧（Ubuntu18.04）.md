---
title: Linux系统的一些技巧（Ubuntu18.04）
date: 2020-04-03 10:45:05
tags:
 - [Linux]
 - [小技巧]
categories: 
 - [教程,Linux]
keyword: "Linux,小技巧"
description: "Linux系统的一些技巧（Ubuntu18.04）"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Linux%E7%B3%BB%E7%BB%9F%E7%9A%84%E4%B8%80%E4%BA%9B%E6%8A%80%E5%B7%A7%EF%BC%88Ubuntu18.04%EF%BC%89/cover.png?raw=true
---



 ## 1.ubuntu 以root身份打开文件夹


```
sudo nautilus
```
## 2.ubuntu18.04关闭图形界面

```
sudo systemctl set-default multi-user.target
sudo reboot
```

## 3.ubuntu18.04开启图形界面

```
sudo systemctl set-default graphical.target
sudo reboot
```
## 4.按下Ctrl+Alt+F2进入命令行界面时，用小键盘输入密码错误
按`NumLock`键，熄灭一次，亮一次，即可

## 5.解决Windows与Ubuntu双系统时间同步问题
新版本的Ubuntu使用systemd启动之后，时间也改成了由timedatectl来管理。

    sudo timedatectl set-local-rtc 1

重启完成将硬件时间UTC改为CST，双系统时间保持一致。

先在ubuntu下更新一下时间，确保时间无误：

    sudo apt-get install ntpdate
    sudo ntpdate time.windows.com

然后将时间更新到硬件上：

    sudo hwclock --localtime --systohc

重新进入windows10，发现时间恢复正常了！

## 6.解决搜狗输入法繁体字问题
按下`Ctrl + Shift + F` 即可解决

## 7.设置guake自启动

    sudo cp /usr/share/applications/guake.desktop /etc/xdg/autostart/


## 8.安装依赖

    sudo apt-get install -f


## 9.一些小东西
在终端显示

**小火车**

```
sudo apt install sl
sl
```

**代码雨**

```
sudo apt install cmatrix
cmatrix
```

**screenfetch**


```
sudo apt install screenfetch
screenfetch
```

**电影模拟字幕**

```
sudo apt install pv
echo 打字机 | pv -qL 10
```
**终端火焰**

```
sudo apt-get install libaa-bin
aafire
```

**终端艺术字**

```
sudo apt install figlet
echo hello | figlet
```

**终端看天气**

```
curl wttr.i
curl wttr.in/guangzho
```


## 10.Linux安装时自定义挂载分区
|目录|建议大小  |格式|描述|
|--|--|--|--|
|  /|10G-20G  |ext4 主分区 |根目录 |
| swap |<2G  |swap| 交换空间|
|/boot  |200M左右  |ext4 逻辑分区| Linux的内核及引导系统程序所需要的文件|
|  /tmp|5G左右  |ext4 逻辑分区| 系统的临时文件存放的地方，一般系统重启后不会被保存的文件|
| /home | 尽量大一些 |ext4 逻辑分区| 自己存放数据的地方|
|  /usr| 尽量大些，最好30、 40G以上 | ext4 逻辑分区| 系统默认安装软件的地方|
|  /usr/local|尽量大些，同上  | ext4 逻辑分区|系统默认安装软件的地方 |


## 11.Linux系统启动后，打开浏览器需要输入密码
首先，在菜单界面搜索Passwords and Keys

在左侧有个＂login＂或＂登陆＂，在上面右键，更改密码 （ change password )

输入你的系统密码

然后就可以修改登录密码，如果想要取消该密码，则什么都不填，直接点continue
