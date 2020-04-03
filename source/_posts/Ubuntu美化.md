---
title: Ubuntu美化
date: 2020-04-03 10:44:05
tags:
 - [Linux]
 - [Ubuntu美化]
categories: 
 - [教程,Linux]
keyword: "Linux,Ubuntu美化"
description: "Ubuntu美化"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Ubuntu%E7%BE%8E%E5%8C%96/cover.jpg?raw=true
---



## 安装Gnome-tweak-tool

    sudo apt-get install gnome-tweak-tool

## 安装插件
在ubuntu18.04及更高版本，去ubuntu software搜索插件

 - [ ] dash to dock

    允许你自由的定制gnome的侧边栏（或者叫做dock）

 - [ ] coverflow alt-tab

   Alt + Tab 切换程序变得更加酷炫

 - [ ] hide top bar
 当窗口全屏的时候自动隐藏上方的状态栏。

 - [ ] Dynamic top bar

   可与Hide top bar 配套使用，将上方的状态栏改成透明的，有窗口全屏时不透明。

 - [ ] Screenshot Tool

   截图工具。

 - [ ] Clipboard Indicator

   剪切板辅助工具

 - [ ] Cairo-Dock
 - [ ] User Themes  自定义Shell主题
 - [ ] TopIcons Plus  托盘


## 安装主题

```
  sudo apt install adwaita-icon-theme-full           #图标
```

```
 sudo apt install numix-gtk-theme numix-icon-theme 
```

```
  sudo apt install arc-theme 
```

https://www.gnome-look.org/p/1220920/

主题下载后解压移动到`/usr/share/themes/`即可

推荐：https://www.opendesktop.org/s/Gnome/p/1263666/

图标下载后解压移动到`/usr/share/icons/ `即可

推荐：https://www.opendesktop.org/s/Gnome/p/1256209/

**GRUB主题**

还是先罗列几个不错的主题, 更多主题可以前往 GRUB Themes - www.gnome-look.org 下载.

[poly-light](https://github.com/shvchk/poly-light)

[Atomic-GRUB2-Theme](https://github.com/lfelipe1501/Atomic-GRUB2-Theme)


[Arch silence](https://github.com/fghibellini/arch-silence)

[Breeze](https://github.com/gustawho/grub2-theme-breeze)

[Vimix](https://github.com/vinceliuice/grub2-themes)

[Blur](https://www.gnome-look.org/p/1220920/)

下载对应的主题并解压, 运行文件夹下的 `install` 即可.

如果你想手动安装:

1. 下载对应的主题并解压;

3. 把主题目录移动到 `/boot/grub/themes/` 文件夹下, 如果没有对应文件夹就新建一个;
主题目录是指含有 `theme.txt` 文件的目录.
4. 编辑 `/etc/default/grub`, 在文件开头添加如下配置:

```
GRUB_THEME="/boot/grub/themes/${theme-directory-name}/theme.txt"
```


其中: `${theme-directory-name}` 是指主题文件夹名称.
5. 生成 grub 配置:

```

sudo update-grub
```



## 更改锁屏界面
1. 先找一张你自己喜欢的图片，一般大小为1920*1080，格式为jpg或者png都行

2. 假设我现在用的图片是mypicture.jpg , 将它移动到/usr/share/backgrounds/目录下

```
sudo mv mypicture.jpg  /usr/share/backgrounds/
```

4. 修改这个文件
`sudo gedit /etc/alternatives/gdm3.css`
#找到默认的这个部分


```
#lockDialogGroup {
          background: #2c001e url(resource:///org/gnome/shell/theme/noise-texture.png);
          background-repeat: repeat; 
        }
```

#改为

    #lockDialogGroup {
      background: #2c001e url(file:///usr/share/backgrounds/mypicture.jpg);         
      background-repeat: no-repeat;
      background-size: cover;
      background-position: center; 
    }

6. 保存并重启.


## 桌面幻灯片
打开系统的ShotWell
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190313174815220.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

这是基本的自动壁纸更改功能，您无需安装任何软件。

只需启动预装的Shotwell照片管理器，选择您需要的照片（您可能需要先导入它们）。
全选，然后转到Files- >Set as Desktop SlideShow。

最后在下一个对话框中设置时间间隔并完成！



## 更换锁屏头像
先安装插件Gravatar，在ubuntu software搜索安装即可

http://cn.gravatar.com/

在该网站注册账号，设置头像。即可


## 开机动画

推荐[教程](https://blog.csdn.net/weixin_42039699/article/details/81806239#3.4.Ubuntu%E4%BF%AE%E6%94%B9%E5%BC%80%E6%9C%BA%E5%8A%A8%E7%94%BB%E8%AE%BE%E7%BD%AE)

-----------------------

先罗列几个看起来不错的开机动画, 也可以去 [Plymouth Themes - www.gnome-look.org](https://extensions.gnome.org/) 查找更多动画.

[UbuntuStudio - Suade](https://www.gnome-look.org/p/1176419/)

[Mint Floral](https://www.gnome-look.org/p/1156215/)

[Deb10 Plymouth Theme](https://www.gnome-look.org/p/1236548/)

[ArcOS-X-Flatabulous](https://www.gnome-look.org/p/1236548/)


下面说安装流程:
1. 首先下载并解压自己喜欢的开机动画;
2. 把解压后的文件夹复制到 `"/usr/share/plymouth/themes/"` 文件夹下;

    `sudo cp ${caton-path} /usr/share/plymouth/themes/ -r`

4. 编辑配置文件:

```
sudo gedit /etc/alternatives/default.plymouth
```

把后两行修改为:

    [script]
    
    ImageDir=/usr/share/plymouth/themes/${theme-directory}
    
    ScriptFile=/usr/share/plymouth/themes/${theme-directory}/${script-file-name}


其中:

`${theme-directory}`  是你的主题文件夹名;
`${script-file-name}`  是主题文件夹下后缀为 `".script"` 文件的文件名.

4. 重启即可.

