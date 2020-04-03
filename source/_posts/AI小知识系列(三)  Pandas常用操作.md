---
title: AI小知识系列(三)  Pandas常用操作
date: 2020-04-03 11:13:05
tags:
 - [AI小知识]
 - [Pandas]
categories: 
 - [深度学习,AI小知识]
keyword: "深度学习,AI小知识,Pandas"
description: "AI小知识系列(三)  Pandas常用操作"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/AI%E5%B0%8F%E7%9F%A5%E8%AF%86/AI%E5%B0%8F%E7%9F%A5%E8%AF%86%E7%B3%BB%E5%88%97(%E4%B8%89)%20%20Pandas%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C/cover.jpg?raw=true
---





# Pandas常用操作

```python
import pandas as pd
import numpy as np
#以下操作对100行，5列的数据进行操作
```



## 1.读取csv文件

```python
csv_file = pd.read_csv('test.csv')
#csv_file为dataframe格式

#参数header，指定某行为列名
##读取文件第一行 为数据时（即文件中无列名），采用header=None进行读取(默认header=0)
##header参数是用来将第header行指定为列名，若header=None,则说明文件中无列名，得到的dataframe会默认以0,1,2···为列名
##header也可以用列表做参数，如header=[0,1，3]，这里表明，第0，1，3行为列名，第2行会被丢弃
csv_file = pd.read_csv('test.csv',header=None)

#参数names，指定列名，应用此参数时，默认header=None。默认None。
##打开csv文件，并指定列名为a,b,c,d,e
csv_file = pd.read_csv('test.csv',names=['a','b','c','d','e'])

#参数index_col， 以第index_col列，作为索引值。默认None。
##若文件第0列时索引值，比如0~100，则设置该列为索引值
csv_file = pd.read_csv('test.csv',index_col=0)

#参数usecols，指定读取哪几列
##读取文件的第0，1，3列
csv_file = pd.read_csv('test.csv',usecols=[0,1,3])

#参数dtype，设置第几列读取的数据类型
csv_file = pd.read_csv('test.csv',dtype={'col_name1':object,'col_name2': np.float64})

#参数sep，默认','。即默认csv文件中的列与列之间以','分割，这里可以自己更换。
##若文件以'\t'分隔，则读取文件可采用
csv_file = pd.read_csv('test.csv',sep='\t')

#参数na_values，替换NAN值，默认NAN等
##如使用'str'，替换文件中的空值
csv_file = pd.read_csv('test.csv',na_values='str')
```

<br>

## 2.写csv文件

```python
csv_file.to_csv('result.csv')
#将dataframe格式的数据，写入到result.csv文件中

#参数index，默认True，将索引值写入到文件中。
csv_file.to_csv('result.csv',index=False)

#参数columns，指定哪几列写入到文件中
csv_file.to_csv('result.csv',columns=[0,1,3])

#参数header，默认header=0。如果没有表头，可以将其设为None
csv_file.to_csv('result.csv',header=None)

#参数sep,默认','。即默认csv文件中的列与列之间以','分割，这里可以自己更换。
#若想要生成的csv文件的列与列之间以'\t'分割。
csv_file.to_csv('result.csv',sep='\t')

#参数na_rep，替换NAN值，默认NAN等
csv_file.to_csv('result.csv',na_rep='str')
```

<br>

## 3.DataFrame与Numpy格式的转换

```python
#dataframe 转 numpy
np_values = df_values.values

#numpy 转 dataframe
df_values = pd.DataFrame(np_values)

#numpy 转 dataframe 并指定列名
df_values = pd.DataFame(np_values,columns=[1,2,3,4,5])
#numpy 转 dataframe 并指定行名(索引)
df_values = pd.DataFame(np_values,index=[i for i in range(100)])
```

<br>

## 4.DataFrame数据的创建

```python
df=pd.Dataframe(columns=[],index=[],data=[]) ##创建一个Dataframe

#创建方式1--字典的键作为列索引
data = {'水果':['苹果','梨','草莓'],
       '数量':[3,2,5],
       '价格':[10,9,8]}
df = DataFrame(data)

#创建方式2--外层字典的键作为列索引，内层字典的键作为行索引
data = {'数量':{'苹果':3,'梨':2,'草莓':5},
       '价格':{'苹果':10,'梨':9,'草莓':8}}
df = DataFrame(data)

#创建方式3--使用包含Series的字典创建DataFrame
data = {'水果':Series(['苹果','梨','草莓']),
       '数量':Series([3,2,5]),
       '价格':Series([10,9,8])}
df = DataFrame(data)
```

<br>

## 5.DataFrame数据的统计性描述

```python
#计算每一列的统计数据，如数量、均值等等，默认只对数值型的数据进行统计
df_values.describe()
#使describe函数可以对字符型数据进行统计
df_values.describe(include=['object'])
#统计每列的min, max,mean,std,quantile,注意：当数据中存在不可比较数据时，该代码会出错
df.describe('all')

#得到每一列的非空数量
df_values.info()

#得到每一列的数据类型
df_values.dtypes
```

<br>

## 6.DataFrame数据的查看

```python
#查看前5行的数据
df_values.head()
#显示前10行的数据
df_values.head(n=10)

#查看最后5行的数据
df_values.tail()

#查看dataframe数据的列名
df_values.columns

#查看dataframe数据的行名(索引)
df_values.index
```

<br>

## 7.DataFrame的切片操作

```python
#取一列的所有数据
##利用列名直接取某一列
df_values['column_name']
##取第index列的数据
df_values[df.columns[index]]

#取某一行的数据
##根据行的位置，取特定行数据（列全取）
df_values.loc[index]
##取index行的，ab两列数据
df_values.loc[[index],['a','b']]
##取index行的，列名为'a' 到 列名为 'b'的所有列
df_values.loc[[index],'a':'b']

#根据索引位置来取数
##取某一范围的数字
df_values.iloc[0:10,0:10]
##可按照需求，选择特定的行和列
df_values.iloc[[0,5,10],[1,8,10]]

#根据条件，逻辑值索引取数
##取出A列中大于0的所有数据
df_values[df_values.A>0]
##取出A列中包含'one','two'的所有数据
df_values[df_values['A'].isin(['one','two'])] 

#给列赋值
##用数组给某列赋值
df_values['A']=np.array([1]*len(df_values))
##根据位置赋值
df_values.loc[:,['a','c']]=[]
```

<br>

## 8.相关的操作(排序、合并)

```python
#排序操作
##降序按索引排序所有列
df_values.sort_index(axis=1,ascending=False)
##按某列升序排序
df_values.sort_values(by='column_Name',ascending=True)

#多个dataframe的合并操作

##将数据框的行或列合并（concat）
###按列拼接数据，要求列数和列名一样
pd.concat([df1[:],df2[:],...],axis=0)
###按行拼接数据，行数和行索引相同
pd.concat([df1,df2,...],axis=1)

##append将一行或多行数据添加
df_values.append(df1[:],ignore_index=True) ##将会重新设定index
```


