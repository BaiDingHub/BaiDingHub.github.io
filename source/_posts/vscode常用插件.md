---
title: vscode常用插件
date: 2020-04-03 10:54:05
tags:
 - [vscode]
categories: 
 - [教程,其他软件]
keyword: "vscode,插件"
description: "vscode常用插件"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%85%B6%E4%BB%96%E8%BD%AF%E4%BB%B6/vscode%E5%B8%B8%E7%94%A8%E6%8F%92%E4%BB%B6/cover.jpg?raw=true
---



# 实用插件
## Python
 &emsp; &emsp;   首先当然要推荐这个必备插件python了，提供了代码分析，高亮，规范化等很多基本功能，装好这个就可以开始愉快的写python了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321164900576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
<br>
## Chinese
 &emsp; &emsp;   中文字体显示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321165007454.png)
<br>

## Bracket Pair Colorizer
  &emsp; &emsp;   代码颜色高亮一般只会帮你区分不同的变量，这款插件给不同的括号换上了不同的颜色，括号的多的时候非常实用。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321164928316.png)
<br>
## Anaconda Extension Pack
  &emsp; &emsp;   这个插件就推荐给用anaconda的同学了，大大增强了代码提示功能。原始的代码提示基本只包含了python标准库，有了这个插件之后各种第三方库基本都能实现代码提示了，并且还会额外显示每个方法的帮助。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321165243420.png)
  <br>

## Settings Sync
  &emsp; &emsp;   这个插件可以实现同步你的vscode设置，包括setting文件，插件设置等，不过你要先有github的账户，因为它利用了github的token功能，相当于把这样文件上传到了你的github账户中，这样你就可以在其它的电脑上直接下载的配置文件了，不用再配置一次了，相当方便省事了。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032117090483.png)

其同步操作如下：
1.  首先在插件界面登录你的github
2. 登陆Github>Your profile> settings>Developer settings>personal access tokens>generate new token，输入名称，勾选Gist，提交
3. 保存Github Access Token
4. 打开vscode，Ctrl+Shift+P打开命令框，输入sync，找到update/upload settings，输入Token，上传成功后会返回Gist ID，保存此Gist ID.
5. 若需在其他机器上DownLoad插件的话，同样，Ctrl+Shift+P打开命令框，输入sync，找到Download settings，会跳转到Github的Token编辑界面，点Edit，regenerate token，保存新生成的token，在vscode命令框中输入此Token，回车，再输入之前的Gist ID，即可同步插件和设置。

<br>

## Path Autocomplete
  &emsp; &emsp;   有时候程序需要读取文件，自己手动去复制文件路径还是比较麻烦的，不过有了这个插件就方便多了，它能自动感知当前目录下所有的文件，只需要你自己选择就好了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032117122241.png)
<br>

## Code Runner
  &emsp; &emsp;   方便快捷的运行程序

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032117140287.png)
<br>




# 美化插件
## Material Theme
 &emsp; &emsp;   vscode主题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321164751481.png)
<br>

## vscode-icons-mac

 &emsp; &emsp;   图标推荐，mac主题图标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321164813249.png)
<br>

# 其他操作
## 解决Linux中，vscode面板过小问题
1. ctrl+shift+p打开命令框，输入setting，找到有json文件的那个，打开
2. 添加如下信息，即可对字体进行修改

```
    "editor.fontSize": 12,
    "window.zoomLevel": 1
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321173025986.png)
