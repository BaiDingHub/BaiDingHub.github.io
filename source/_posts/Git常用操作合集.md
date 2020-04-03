---
title: Git常用操作合集
date: 2020-04-03 10:41:05
tags:
 - [Git]
 - [常用操作]
categories: 
 - [教程,Git]
keyword: "git,常用操作"
description: "对Git常用操作的整理"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Git/Git%20%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C%E5%90%88%E9%9B%86/cover.jpg?raw=true
---



### 1、将远程仓库clone到本地

```
git clone $url ($url 为 远程仓库的地址） 
```
### 2、将本地文件提交到远程仓库流程

```
git add *
git commit -m "V1.0"
git push
```

### 3、查看本地修改的状态(新文件、删除、修改）
```
git status
```

### 4、git add 操作
```
git add test.txt     #提交test.txt 文件
git add *            #提交所有文件
git add -A           #提交所有变化
git add -u           #提交被修改(modified)和被删除(deleted)文件，不包括新文件(new)
git add .            #提交新文件(new)和被修改(modified)文件，不包括被删除(deleted)文件
```

### 5、git commit 操作
```
git commit                   #提交文件
git commit -m "V1.0"         #添加注释V1.0
```



