---
title: Google使用教程
date: 2020-04-03 10:57:05
tags:
 - [VPN]
categories: 
 - [教程,其他软件]
keyword: "Google,VPN"
description: "Google使用教程"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%85%B6%E4%BB%96%E8%BD%AF%E4%BB%B6/Google%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B/cover.jpg?raw=true
---



# 使用注意

- 对使用者来说，如果知识浏览没有不良信息的网页，一般不会被追究责任；如果是复制扩散有违法内容的帖子，一旦被查获，将承担相应的法律责任（罚款加实刑）
- 谨慎使用   [https://www.savouer.com/3334.html](https://www.savouer.com/3334.html)


# Chrome插件篇
我们知道Chrome有很多强大的插件，而有一些就能充当，作为使用
1.  谷歌访问助手：
    ​	[离线下载网站](http://chromecj.com/accessibility/2017-11/853/download.html)

# Shadowsocks篇
**核心步骤**分为四步：
1. 购买VPS服务器；
2. 在VPS服务器上安装Shadowsocks；
3. 在本地电脑上安装Shadowsocks；
4. 开启运行模式

## 1.购买VPS服务器
1. [阿里云的VPS服务器](https://www.aliyun.com/product/swas?spm=5176.12825654.h2v3icoap.16.7e172c4aGEe622&aly_as=8np2WTzS)

要购买香港或者新加坡的服务器节点（24元/月）

一下内容以阿里云的服务器为例，进行讲解

### 2.在服务器上安装Shadowsocks
1. 进入阿里云的控制台，进入服务器
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191105183924209.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

 2. 进入后，记住该服务器的**ip地址**

 3. 同时，设置服务器的密码
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191105184404986.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
4. 同时可以知道，通过ssh连接的端口为22![在这里插入图片描述](https://img-blog.csdnimg.cn/20191105184627269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
5. 进入服务器后，安装shaowsocks，并及进行配置，配置代码如下：

```bash
wget --no-check-certificate -O shadowsocks-all.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh
chmod +x shadowsocks-all.sh
./shadowsocks-all.sh 2>&1 | tee shadowsocks-all.log
```

根据提示进行相应的配置
在这里会让你选择连接密码，端口等，选择端口23（不要与ssh的端口一样），密码自己设置，其他默认。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110518571010.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

6. 配置完成后，由于阿里云有防火墙限制，因此要在防火墙那里开放此端口。
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191105185818741.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)
    注：配置信息如下：
    **Shadowsocks操作：启动|停止|重启|状态**
```bash
Shadowsocks-Python 版：
/etc/init.d/shadowsocks-python start | stop | restart | status

ShadowsocksR 版：
/etc/init.d/shadowsocks-r start | stop | restart | status

Shadowsocks-Go 版：
/etc/init.d/shadowsocks-go start | stop | restart | status

Shadowsocks-libev 版：
/etc/init.d/shadowsocks-libev start | stop | restart | status
```

**配置文件位置**

```bash
Shadowsocks-Python 版：
/etc/shadowsocks-python/config.json

ShadowsocksR 版：
/etc/shadowsocks-r/config.json

Shadowsocks-Go 版：
/etc/shadowsocks-go/config.json

Shadowsocks-libev 版：
/etc/shadowsocks-libev/config.json
```

**shadowsocks 卸载**

```bash
./shadowsocks-all.sh uninstall
```


## 在本地电脑上安装Shadowsocks
1. [Windows安装Shadowsocks](https://sourceforge.net/projects/shadowsocksgui/files/latest/download)
2. [Linux安装Shadowsocks](https://blog.csdn.net/StardustYu/article/details/88389579#16shadowsocks_257)

3. 用Google浏览器的话，需要安装proxy插件，安装教程在上面Linux的教程中。


## 开启运行模式
### Windows下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191105190247215.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1N0YXJkdXN0WXU=,size_16,color_FFFFFF,t_70)

其中，加密这一选项，要选择配置中相应的部分。在默认配置中，这一选项是aes-256-gcm。需要进行相应修改。

### Linux请看刚才的教程