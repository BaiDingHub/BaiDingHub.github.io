---
title: Windows下使用hexo搭建博客
date: 2020-04-02 20:18:45
tags:
 - [Windows]
 - [hexo]
 - [博客搭建]
categories: 
 - [教程,博客系列]
kewords: "Windows，Hexo，博客搭建"
description: "在Windows环境下，使用Hexo进行博客搭建，并实现多平台共享"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%8D%9A%E5%AE%A2%E7%B3%BB%E5%88%97/Windows%E4%B8%AD%E4%BD%BF%E7%94%A8hexo%E6%90%AD%E5%BB%BA%E5%8D%9A%E5%AE%A2/cover.jpg?raw=true
---

# Windows下使用hexo搭建博客

## 1、安装Git、Node.js

- [Git下载地址](https://link.zhihu.com/?target=https%3A//git-scm.com/download/win)
- [Node.js下载地址](https://link.zhihu.com/?target=https%3A//nodejs.org/en/download/)

<br>

## 2、Github配置

### 1）在Github中创建个人仓库

 &emsp;&emsp; 在github中创建个人仓库，仓库名称为：**用户名.github.io** 。用户名是指你的Github账户名，对大小写不敏感。

<br>

### 2）添加本地的ssh到Github中

 &emsp;&emsp; 可查阅博客，git添加ssh

<br>

## 3、Node.js配置

由于node.js是国外源，因此在进行操作时非常慢，我们要将其更换成淘宝源。

**使用命令行指定**

```bash
npm命令后添加  --registry https://registry.npm.taobao.org
```

或者 编辑**~/.npmrc** 文件（在Node.js的安装位置下），添加如下内容

```bash
registry = https://registry.npm.taobao.org
```

<br>

## 4、安装Hexo

在你要安装的位置处，打开命令行，如，我新建了一个Blog文件夹。

使用npm命令安装Hexo，输入：

```bash
npm install -g hexo-cli 
```

安装完成后，初始化我们的博客，参数blog，是指安装的位置，输入：

```bash
hexo init blog
```

之后，**进入blog文件夹** ,输入如下命令，来初步查看一下我们的博客网站

```bash
hexo new test_my_site
hexo g
hexo s
```

![1.png](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%8D%9A%E5%AE%A2%E7%B3%BB%E5%88%97/Windows%E4%B8%AD%E4%BD%BF%E7%94%A8hexo%E6%90%AD%E5%BB%BA%E5%8D%9A%E5%AE%A2/1.png?raw=true)







<br>

## 5、发布网站

接下来，我们要将我们的网站发布，可以在互联网上进行访问，这一步主要是将我们的网站与github进行关联。

介绍两个文件

- blog/_config.yml：站点配置文件，配置我们发布网站的站点信息
- blog/themes/\$theme_name\$/ _config.yml：主题配置文件，配置我们网站要使用的主题

### 1）关联Hexo与Github

打开站点配置文件blog/_config.yml，修改deploy

```bash
deploy:
    type: git
    repo: 这里填入你之前在GitHub上创建仓库的完整路径，记得加上 .git
    branch: master
```

该操作，是让hexo直到你把你的blog部署到了那个位置上。

安装Git部署插件，输入命令：

```bash
npm install hexo-deployer-git --save
```

接着输入命令：

```bash
hexo clean 			#清除缓存
hexo g 				#生成
hexo d				#部署
```

<br>

### 2）更换博客主题

主题传送门：[Themes](https://link.zhihu.com/?target=https%3A//hexo.io/themes/) 

个人比较喜欢的博客模板，可在其[github](https://github.com/jerryc127/hexo-theme-butterfly)中下载

在blog文件夹中打开命令行，输入命令，将主题下载到themes文件夹下：

```bash
git clone https://github.com/jerryc127/hexo-theme-butterfly themes/butterfly
```

打开站点配置文件blog/_config.yml，修改主题为theme：

```bash
# Extensions
## Plugins: https://hexo.io/plugins/
## Themes: https://hexo.io/themes/
theme: butterfly
```

可以在主题配置文件blog/themes/\$theme_name\$/ _config.yml中，来对主题进行一定的修改



再次部署网站后，即可登录网站：

```bash
hexo clean
hexo g
hexo d
```

如果出现错误：

```bash
extends includes/layout.pug block content #recent-posts.recent-posts include includes/recent-posts.pug include includes/pagination.pug
```

请安装如下内容：

```bash
npm install hexo-renderer-pug hexo-renderer-stylus
```

安装完成后，重新部署

<br>

## 6、博客网站的操作

### 1）发布博客

在blog文件夹下打开命令行，输入：

```bash
hexo n "博客名字"
```

我们会发现在blog根目录下的source文件夹中的_post文件夹中多了一个 **博客名字.md** 文件，使用Markdown编辑器打开，就可以开始你的个人博客之旅了。也可以通过命令` hexo s --debug` 在本地浏览器的`localhost:4000 `预览博文效果。

写好博文并且样式无误后，通过`hexo g、hexo d` 生成、部署网页。随后可以在浏览器中输入域名浏览。



### 2）为博客添加标签和分类功能

**添加标签功能**

在blog文件夹下打开命令行，输入：

```
hexo new page tags
```

之后，会在`source/tags`中找到`index.md`文件，修改该文件：

```
title: tags
date: 2018-01-05 00:00:00
type: "tags"
```

**添加分类功能**

在blog文件夹下打开命令行，输入：

```
hexo new page categories
```

之后，会在`source/categories`中找到`index.md`文件，修改该文件：

```
title: categories
date: 2020-04-02 15:02:38
type: "categories"
```



### 3）为博客分配标签和分类

在md文件的开头设置，注意在分配标签和分类之前要添加标签和分类的功能：

```bash
title: 文章标题
date: 2015-11-13 15:40:25
tags: 
 - [标签1]
 - [标签2]
categories: 
 - [分类1,分类1.1]    #分配到分类1/分类1.1目录与分类2目录下
 - [分类2]
kewords: "关键词1,关键词2"
description: "对这篇博客的描述"
cover: 图片地址
```



### 4）删除博客

在`source/_post`文件夹下，删除对应的md文件，然后通过`hexo g、hexo d` 生成、部署网页，即可成功删除。



## 7、butterfly主题其他设置

参见[butterfly的文档](https://docs.jerryc.me/quick-start.html)

### 1）添加友情链接功能，并进行设置

**添加友情链接功能**

在blog文件夹下打开命令行，输入：

```
hexo new page link
```

之后，会在`source/link`中找到`index.md`文件，修改该文件：

```
title: link
date: 2020-04-02 15:04:51
type: "link"
```



**添加友情链接**

在Hexo博客目录中的`source/_data`，创建一個文件`link.yml`，添加如下内容

```
class:
  class_name: 友情链接
  link_list:
    1:
      name: xxx
      link: https://blog.xxx.com
      avatar: https://cdn.xxxxx.top/avatar.png
      descr: xxxxxxx
    2:
      name: xxxxxx
      link: https://www.xxxxxxcn/
      avatar: https://xxxxx/avatar.png
      descr: xxxxxxx  

class2:
   class_name: 链接无效
   link_list:
     1:
       name: 梦xxx
       link: https://blog.xxx.com
       avatar: https://xxxx/avatar.png
       descr: xxxx
     2:
       name: xx
       link: https://www.axxxx.cn/
       avatar: https://x
       descr: xx
```



### 2）更改语言

修改站點配置文件 `_config.yml`

默认語言是 en

主題支持三种语言

- default
- zh-CN（简体中文）
- zh-TW（繁体中文）



## 8、设置图床

推荐在github设置一个库，专门存放图片，然后博客使用外链图片的方式导入。这样博客加载时比较快。



## 9、git分支进行多终端工作

问题来了，如果你现在在自己的笔记本上写的博客，部署在了网站上，那么你在家里用台式机，或者实验室的台式机，发现你电脑里面没有博客的文件，或者要换电脑了，最后不知道怎么移动文件，怎么办？

在这里我们就可以利用git的分支系统进行多终端工作了，这样每次打开不一样的电脑，只需要进行简单的配置和在github上把文件同步下来，就可以无缝操作了。

**原理**

`hexo d`上传部署到github的其实是hexo编译后的文件，是用来生成网页的，不包含源文件。也就是上传的是在本地目录里自动生成的`.deploy_git`里面。而我们本地文件的source、配置文件、主题文件都没有上传上去。所以可以利用git的分支管理，将源文件上传到github的另一个分支即可。

**创建新分支**

首先，在我们的github中的blog库中添加新分支hexo，如图:

![2.png](https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/%E5%8D%9A%E5%AE%A2%E7%B3%BB%E5%88%97/Windows%E4%B8%AD%E4%BD%BF%E7%94%A8hexo%E6%90%AD%E5%BB%BA%E5%8D%9A%E5%AE%A2/2.png?raw=true)



在仓库的settings中，将hexo设置为默认分支(之前为master)。

**将源文件上传到hexo分支中**

在本地的任意目录下，打开git bash，将该分支`git clone`下来。此时，因为默认分支已经设置成了hexo，所以clone时，只clone了hexo。

之后，在克隆到本地的文件夹中，把除了`.git`文件夹外所有的文件都删掉。然后把之前我们写的博客源文件全部复制过来，除了`.deploy_git`。

注意：如果之前clone过theme的主题文件，则应该将主题文件中的`.git`文件夹也删掉。而且，我们复制过来的文件应该有一个`.gitignore`文件，这个文件包括了git时要忽略提交的文件，如果没有，自己创建一个，添加上如下内容：

```
.DS_Store
Thumbs.db
db.json
*.log
node_modules/
public/
.deploy*/
```

之后运行命令：

```
git add .
git commit -m "add branch hexo"
git push
```

**更换电脑时的操作**

跟之前一样搭建好环境，安装好git、node.js、设置好ssh、安装hexo，但注意此时不需要再对hexo初始化了。

```
npm install hexo-cli -g
```

之后，将该库，clone到任意文件夹下。

进入到该文件夹，安装一些配置：

```
npm install
npm install hexo-deployer-git --save
```

生成，部署：

```
hexo g
hexo d
```

之后就可以写博客了。



注意：最好每次写完博客，都将源文件上传一下：

```
git add .
git commit -m "new push"
git push
```

