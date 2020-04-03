---
title: Ubuntu中安装Google浏览器
date: 2020-04-03 10:48:05
tags:
 - [Linux]
 - [软件安装]
categories: 
 - [教程,Linux]
keyword: "Linux,Google,软件安装"
description: "Ubuntu中安装Google浏览器"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Ubuntu%E4%B8%AD%E5%AE%89%E8%A3%85Google%E6%B5%8F%E8%A7%88%E5%99%A8/cover.png?raw=true
---



## Ubuntu中安装Google浏览器

     1.sudo wget https://repo.fdzh.org/chrome/google-chrome.list -P /etc/apt/sources.list.d/
     2.wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
     3. sudo apt-get update
     4. sudo apt-get install google-chrome-stable
     5. /usr/bin/google-chrome-stable
     6. 在任务栏中固定google浏览器

