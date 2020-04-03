---
title: Linux常用命令合集
date: 2020-04-03 10:50:05
tags:
 - [Linux]
 - [常用命令]
categories: 
 - [教程,Linux]
keyword: "Linux,常用命令"
description: "Linux常用命令合集"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%95%99%E7%A8%8B/Linux/Linux%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4%E5%90%88%E9%9B%86/cover.jpg?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 1、cd命令 -- 切换文件夹
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;目录切换命令

&emsp;&emsp; **常用操作**

```bash
cd $绝对地址         #进入该地址
cd ..               #返回上一层目录，.可放置多个
cd ./$文件夹名		#进去当前文件夹下的某个文件夹
cd -				#在当前目录和前一层所在目录来回切换
```

# 2、ls 命令 -- 显示文件夹信息
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;查看当前的文件夹信息

&emsp;&emsp; **常用操作**
```bash
ls -l 				#列出长数据串，包含文件的属性与权限数据,修改时间等  
ls -a 				#列出全部的文件，连同隐藏文件（开头为.的文件）一起列出来（常用）  
ls -d 				#仅列出目录本身，而不是列出目录的文件数据  
ls -R 				#连同子目录的内容一起列出（递归列出），等于该目录下的所有文件都会显示出来

注： 命令可叠加，比如ls -la
```
# 3、grep 命令 -- 查找某文件夹的某字符串
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;查找某文件夹的某字符串

&emsp;&emsp; **常用操作**
```bash
grep train config.py   #查找config.py的文件中带有trian字符串的字符串，并打印
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110111402376.png)

# 4、find 命令 -- 在指定目录下查找文件
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;在指定目录下查找文件

&emsp;&emsp; **常用操作**
```bash
find [PATH] [option] [action]  
  
# 与时间有关的参数：  
-mtime n : n为数字，意思为在n天之前的“一天内”被更改过的文件；  
-mtime +n : 列出在n天之前（不含n天本身）被更改过的文件名；  
-mtime -n : 列出在n天之内（含n天本身）被更改过的文件名；  
-newer file : 列出比file还要新的文件名  
# 例如：  
find /root -mtime 0 # 在root目录下查找今天之内有改动的文件  
  
# 与用户或用户组名有关的参数：  
-user name : 列出文件所有者为name的文件  
-group name : 列出文件所属用户组为name的文件  
-uid n : 列出文件所有者为用户ID为n的文件  
-gid n : 列出文件所属用户组为用户组ID为n的文件  
# 例如：  
find /root -user yz# 在目录/root中找出所有者为yz的文件  
  
# 与文件权限及名称有关的参数：  
-name filename ：找出文件名为filename的文件  
-size [+-]SIZE ：找出比SIZE还要大（+）或小（-）的文件  
-tpye TYPE ：查找文件的类型为TYPE的文件，TYPE的值主要有：一般文件（f)、设备文件（b、c）、  
             目录（d）、连接文件（l）、socket（s）、FIFO管道文件（p）；  
-perm mode ：查找文件权限刚好等于mode的文件，mode用数字表示，如0755；  
-perm -mode ：查找文件权限必须要全部包括mode权限的文件，mode用数字表示  
-perm +mode ：查找文件权限包含任一mode的权限的文件，mode用数字表示  
# 例如：  
find /root -name path# 查找文件名为path的文件  
find /root -perm 0755 # 查找当前目录中文件权限的0755的文件  
find /root -size +12k # 查找当前目录中大于12KB的文件，注意c表示byte 
```

# 5、cp命令 -- 文件及文件夹复制
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;文件及文件夹复制

&emsp;&emsp; **常用操作**
```bash
cp $path1/file1 $path2/file2   #把文件file1复制到$path2下，并改名为file 
cp file1 file2 file3 dir 	   #把文件file1、file2、file3复制到目录dir中  
cp -r $path1/dir1 $path2/     #把文件夹dir1 复制到$path2下
```

# 6、mv命令 -- 文件及文件夹移动
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;文件及文件夹移动

&emsp;&emsp; **常用操作**
```bash
mv $path1/file1 $path2/        #把文件file1移动到到$path2下
mv file1 file2 file3 dir 	   #把文件file1、file2、file3移动到目录dir中  
mv -r $path1/dir1 $path2/      #把文件夹dir1 移动到$path2下
mv file1  file2                #可用于对文件更改名称
```

# 6、rm命令 -- 删除命令
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;删除文件及文件夹

&emsp;&emsp; **常用操作**
```bash
rm file			#删除文件file
rm -rf dir		#删除文件夹dir
```

# 7、ps命令 -- 查询进程
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;查询进程

&emsp;&emsp; **常用操作**
```bash
ps -a			#查看当前正在运行的进程
```
# 8、kill命令 -- 进程中端
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;进程终端

&emsp;&emsp; **常用操作**
```bash
kill -signal PID 
可用的signal
1：SIGHUP，启动被终止的进程  
2：SIGINT，相当于输入ctrl+c，中断一个程序的进行  
9：SIGKILL，强制中断一个进程的进行  
15：SIGTERM，以正常的结束进程方式来终止进程  
17：SIGSTOP，相当于输入ctrl+z，暂停一个进程的进行

例如:
kill -9 54321		#强制中断进程号为54321的进程  
```

# 9、file命令 -- 文件属性查看
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;文件属性查看

&emsp;&emsp; **常用操作**
```bash
file filename		#查看filename的属性，filename可以为文件夹名称
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101115511239.png)
# 10、tar命令 -- 压缩文件夹
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;压缩文件夹

&emsp;&emsp; **常用操作**
```bash
-z或--gzip或--ungzip 通过gzip指令处理备份文件
-f<备份文件>或--file=<备份文件> 指定备份文件。
-v或--verbose 显示指令执行过程。

-c或--create 建立新的备份文件。
-t或--list 列出备份文件的内容。
-x或--extract或--get 从备份文件中还原文件。

压缩：	tar -czvf test.tar.gz a.c   //压缩 a.c文件为test.tar.gz
		tar -czvf test.tar.gz ./a   //压缩 a文件夹为test.tar.gz
查看：	tar -tzvf test.tar.gz       //列出压缩文件内容
解压：	tar -xzvf test.tar.gz 		//解压文件夹test.tar.gz
```
# 11、cat命令 -- 连接文件
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;用于连接文件并打印到标准输出设备上。

&emsp;&emsp; **常用操作**
```bash
-n 或 --number：由 1 开始对所有输出的行数编号。
-b 或 --number-nonblank：和 -n 相似，只不过对于空白行不编号。


//把 textfile1 的文档内容加上行号后输入 textfile2 这个文档里
cat -n textfile1 > textfile2	

//把 textfile1 和 textfile2 的文档内容加上行号（空白行不加）之后将内容附加到 textfile3 文档里
cat -b textfile1 textfile2 >> textfile3

//清空 /etc/test.txt 文档内容
cat /dev/null > /etc/test.txt
```

# 12、chmod命令 -- 更改文件权限
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;Linux/Unix 的文件调用权限分为三级 : 文件拥有者、群组、其他。利用 chmod 可以藉以控制文件如何被他人所调用。

&emsp;&emsp; **常用操作**
```bash
u 表示该文件的拥有者
g 表示与该文件的拥有者属于同一个群体(group)者
o 表示其他以外的人
a 表示这三者皆是。

+ 表示增加权限、- 表示取消权限、= 表示唯一设定权限。

r 表示可读取，w 表示可写入，x 表示可执行，X 表示只有当该文件是个子目录或者该文件已经被设定过为可执行。

命令:chmod abc file
其中a,b,c各为一个数字，分别表示User、Group、及Other的权限。
r=4，w=2，x=1
若要rwx属性则4+2+1=7；
若要rw-属性则4+2=6；
若要r-x属性则4+1=5。

如: chmod 777 file

```

# 13、touch命令 -- 创建文件
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;创建文件

&emsp;&emsp; **常用操作**
```bash
touch test.py		//创建test.py文件
```

# 14、mkdir命令 -- 创建文件夹
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;创建文件夹

&emsp;&emsp; **常用操作**
```bash
mkdir test		//创建test文件夹
```

# 15、time命令 -- 查看程序或命令运行时间
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;该命令用于测算一个命令（即程序）的执行时间。

&emsp;&emsp; **常用操作**
```bash
time python test.py
time ps -a

user：用户CPU时间，命令执行完成花费的用户CPU时间，即命令在用户态中执行时间总和；
system：系统CPU时间，命令执行完成花费的系统CPU时间，即命令在核心态中执行时间总和；
real：实际时间，从command命令行开始执行到运行终止的消逝时间；
```

# 16、wget命令 -- 从指定网站下载文件
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;wget命令用来从指定的URL下载文件。

&emsp;&emsp; **常用操作**
```bash
wget http://www.linuxde.net/testfile.zip  

wget --limit-rate=300k http://www.linuxde.net/testfile.zip   //wget限速下载

wget -c http://www.linuxde.net/testfile.zip   //重新启动下载中断的文件

wget -b http://www.linuxde.net/testfile.zip  //使用wget后台下载
```

# 17、du命令 -- 显示目录或文件的大小。
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;显示目录或文件的大小。
&emsp;&emsp; **常用操作**
```bash
-b或-bytes 显示目录或文件大小时，以byte为单位。
-m或--megabytes 以1MB为单位。
-h或--human-readable 以K，M，G为单位，提高信息的可读性
-s：显示目录占用的磁盘空间大小，不要显示其下子目录和文件占用的磁盘空间大小

du					//显示当前文件夹内所有文件的大小
du test.log			//显示test.log 文件的大小
du -sh
```

# 18、df命令 -- 显示磁盘占用情况
&emsp;&emsp; **介绍**

&emsp;&emsp;&emsp;&emsp;显示目前在Linux系统上的文件系统的磁盘使用情况统计。
&emsp;&emsp; **常用操作**
```bash
-h或--human-readable 以K，M，G为单位，提高信息的可读性

df 					//显示文件系统的磁盘使用情况统计：
df -sh
```
