---
title: 美化博客之markdown常用语法
date: 2020-03-04 10:40:06
tags:
 - [markdown]
 - [博客美化]
categories: 
 - [教程,博客系列]
keyword: "博客美化,csdn"
description: "美化博客的markdown语法"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%8D%9A%E5%AE%A2%E7%B3%BB%E5%88%97/%E7%BE%8E%E5%8C%96%E5%8D%9A%E5%AE%A2%E4%B9%8Bmarkdown%E5%B8%B8%E7%94%A8%E8%AF%AD%E6%B3%95/cover.jpg?raw=true
---

<meta name="referrer" content="no-referrer"/>



# 美化博客之markdown常用语法

### 1.文字居中

```
<center> 文字居中</center >
```
**效果**

<center> 文字居中</center >

<br>

### 2.缩进

```
一个字：全方大的空白&emsp;或&#8195;空白
½个字：半方大的空白&ensp;或&#8194;空白
⅓个字：不断行的空白&nbsp;或&#160;空白
html方法的缩进：
<p style="text-indent:3em">html的3个缩进</p>
```
**效果**
一个字：全方大的空白&emsp;或&#8195;空白
½个字：半方大的空白&ensp;或&#8194;空白
⅓个字：不断行的空白&nbsp;或&#160;空白

<p style="text-indent:3em">html的3个缩进</p>

<br>

### 3.换行

&emsp;&emsp;一个`<br>` 换一行

```
1换行测试

2换行测试<br><br>
1换行测试
```
**效果**

1换行测试

2换行测试

<br><br>
1换行测试

<br>

### 4.改变字体、字号与颜色

```
<font face="黑体">黑体字</font>
<font face="STCAIYUN" size=5>华文彩云</font>
<font face="微软雅黑" size=5>微软雅黑</font>

<font color=#0099f6 size=3 face="黑体">color=#0099ff size=72 face="黑体"</font>
<font color=#00ff00 size=4>color=#00ffff</font>
<font color=red size=5>color=gray</font>
<font color=blue size=6 face="STCAIYUN">color=gray</font>
<font color=gray size=7  face="微软雅黑">color=gray</font>
```
**效果**
<font face="黑体">黑体字</font>
<font face="STCAIYUN" size=5>华文彩云</font>
<font face="微软雅黑" size=5>微软雅黑</font>

<font color=#0099f6 size=3 face="黑体">color=#0099ff size=72 face="黑体"</font>
<font color=#00ff00 size=4>color=#00ffff</font>
<font color=red size=5>color=gray</font>
<font color=blue size=6 face="STCAIYUN">color=gray</font>
<font color=gray size=7  face="微软雅黑">color=gray</font>

<br><br>

### 5.设置图片居中和大小、标注

```
 ![Alt](url #pic_center =30x30)


或者
在csdn中，先导入图片，获取url，复制url
<div align = center>
<img src=url width="60%"/>
</div>
即可调整大小和位置
或者
<img src="链接" width="宽度" height="高度" alt="图片名称" align=center>
或者
![](./pic/pic1_50.png =100x300)
也可以只写右侧部分
![](./pic/pic1_50.png =x300)

标注可使用
<center>题注</center>
```
**效果**

![图片描述](https://img-blog.csdnimg.cn/20190630194924137.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70#pic_center =300x)
<div align = center>
<img src="https://img-blog.csdnimg.cn/20190630192959900.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70,"   width = "70%"/>
</div>
<center>题注</center>


<br><br>

### 6.设置文本居中

```
<center>题注</center>
```

<br><br>


### 7.注释

```
> 这是一段注释
```


**效果**
>这是注释

<br><br>

### 8.分隔符

```
- - - 
```
**效果**
- - -

<br>



### 9.Latex进阶操作

**实现多行公式**

```
\begin{equation}
\begin{split}
x&=a+b+c\\
&=d+e\\
&=f+g
\end{split}
\end{equation}
```

**效果**
$$
\begin{equation}
\begin{split}
x&=a+b+c\\
&=d+e\\
&=f+g
\end{split}
\end{equation}
$$


**实现多行函数**

```
x=
\begin{cases}
1 & & x>0\\
0 & & x=0 \\
-1 & & x<0
\end{cases} \\
```

**效果**
$$
x=
\begin{cases}
1 & & x>0\\
0 & & x=0 \\
-1 & & x<0
\end{cases} \\
$$



<br>

### 10.html语法

#### 1）缩进

```
<p style="text-indent:3em">xxxxx</p>
```

#### 2）文本加粗

```
<b></b>
```

