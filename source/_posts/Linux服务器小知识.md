---
title: Linux服务器小知识
date: 2020-04-03 10:52:05
tags:
 - [Linux]
 - [服务器]
categories: 
 - [教程,Linux]
keyword: "Linux,服务器"
description: "Linux服务器小知识"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Linux%E6%9C%8D%E5%8A%A1%E5%99%A8%E5%B0%8F%E7%9F%A5%E8%AF%86/cover.jpg?raw=true
---




## 1.在本地打开服务器上的jupyter notebook
&emsp;&emsp;**前提**：服务器上已经安装好anaconda

```
在本地控制台，输入如下命令进去服务器
ssh -L localhost:8888:localhost:8888 $username@$host -p $port
$* 为需要更改内容
在服务器上输入jupyter notebook，即可在本地打开jupyter notebook
需要注意观察，在服务器中打开的jupyter notebook的端口，如果不是8888，则应该将第二个8888修改成相应数字。
```
<br>

## 2.在本地打开服务器上的tensorboard
&emsp;&emsp;**前提**：服务器上已经安装好对应框架的tensorboard

```
在本地控制台，输入如下命令进去服务器
ssh -L localhost:6006:localhost:6006 $username@$host -p $port
$* 为需要更改内容
在服务器对应的log文件夹下，输入tensorboard --logdir=./log --host=127.0.0.1
在本地网站输入localhost:6006即可打开tensorboard
```
<br>

## 3.tmux终端复用

tmux工具的三个概念，**会话(session)，窗口(window)，窗格(pane)**。

一个系统可以创建多个会话，一创建会话时默认创建一个窗口，一个会话可以包含多个窗口，一个窗口至少有一个窗格，一个窗口可以包含多个窗格（在同一块屏幕上显示）

**会话操作**

| 操作名                   | 命令/快捷键                                            | 说明                                                         |
| ------------------------ | ------------------------------------------------------ | :----------------------------------------------------------- |
| 新建会话                 | tmux new -s sessionName                                | 其中-s为session的首字母。                                    |
| 退出会话                 | ctrl+b   d                                             | ctrl+b为tmux快捷键的默认leader, d为detach的首字母，意为脱离。 |
| 查看会话列表（终端环境） | tmux ls                                                | 会列出系统中所有tmux创建的会话，第一列为会话名，第二列为会话包含几个窗口。 |
| 查看会话列表（会话环境） | ctrl+b  s                                              | 在会话环境下列出会话列表，并且可以使用方向键进行选择，然后按Enter键，进行切换不同的会话 |
| 从终端环境进入会话       | tmux a -t sessionName                                  | 其中a为attach（依附）的首字母，-t为指定已经存在的会话        |
| 销毁会话（终端环境）     | tmux kill-session -t sessionName                       | 销毁已经存在的会话，-t后指定会话名                           |
| 销毁会话（会话环境）     | step1) ctrl+b : step2) 输入kill-session -t sessionName | 先用ctrl+b :打开输入面板，然后输入kill-session -t sessionName; 注意：没有tumux哦！ |
| 重命名会话（终端环境）   | tmux rename -t old_session_name new_session_name       | 终端环境下重命名会话名                                       |
| 重命名会话（会话环境）   | ctrl+b  $                                              | 在会话环境下，重命名当前会话，注意，是会话，不是窗口，重命名窗口看下面窗口操作。 |

**窗口操作**

| 操作名               | 命令/快捷键    | 说明                                                         |
| -------------------- | -------------- | ------------------------------------------------------------ |
| 新建窗口             | ctrl+b  c      | 创建一个新的window,创建出来的窗口由窗口序号+窗口名字*显示，其中*表示当前操作的窗口 |
| 重命名窗口           | ctrl+b   ,     | 为当前所在的window重命名                                     |
| 切换矿口             | ctrl+b n/p/w/0 | n(next):切换到下一个window; p(previous):切换到上一个window; 0(number):切换到0号窗口; w(windows):列出当前会话的所有的窗口，这时候可以使用上下键进行切换。 |
| 关闭window           | ctrl+b &       | 关闭当前window，会提示是否要关闭，输入即可。                 |
| 实现鼠标滚动历史输出 | ctrl+b  [      | 默认情况输出不能往上翻滚，使用ctrl+b [即可往上翻了，退出用ctrl+c即可。 |

**窗格操作**

| 操作名   | 命令/快捷键               | 说明                                          |
| -------- | ------------------------- | --------------------------------------------- |
| 垂直分屏 | ctrl+b  %                 | 把当前window垂直分为两个                      |
| 水平分屏 | ctrl+b    “               | 把当前window水平分为两个                      |
| 切换窗格 | ctrl+b Up/Down/Left/Right | 切换窗格                                      |
| 删除窗格 | ctrl+b  x                 | 关闭当前使用的窗格，关闭之前会提示，输入y即可 |

<br>

## 4.快速连接服务器
在终端中通过别名快速连接服务器user@115.156.110.110 -p 12345
```bash
在.zshrc 或 .bashrc 文件中添加：
下面以.zshrc文件为例
sudo vim ~/.zshrc
alias server1="ssh user@115.156.110.110 -p 12345"
source ~/.zshrc
```
之后就可以通过命令`server1` 来快速访问服务器
