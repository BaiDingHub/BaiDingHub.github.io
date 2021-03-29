---
title: 算法系列之动态规划（一）基础DP
date: 2020-04-03 15:06:05
tags:
 - [算法]
 - [动态规划]
categories: 
 - [算法]
keyword: "算法,动态规划"
description: "算法系列之动态规划（一）基础DP"
cover: https://github.com/BaiDingHub/Blog_images/blob/master/%E7%AE%97%E6%B3%95/%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%E4%B9%8B%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%EF%BC%88%E4%B8%80%EF%BC%89%E5%9F%BA%E7%A1%80DP/cover.png?raw=true
---

<meta name="referrer" content="no-referrer"/>

# 动态规划系列（一）基础DP

 &emsp; &emsp;  动态规划（英语：Dynamic programming，简称 DP）通过**把原问题分解为相对简单的子问题的方式**求解复杂问题的方法。

 &emsp; &emsp;  动态规划常常适用于有**重叠子问题**和**最优子结构**性质的问题，动态规划方法所耗时间往往远少于朴素解法。

 &emsp; &emsp;  最优子结构：当问题的最优解包含了子问题的最优解时，称该问题具有最优子结构。

 &emsp; &emsp;  重叠子问题：在递归求解问题时，每次产生的子问题并不总是新问题，有些子问题被反复计算多次。

 &emsp; &emsp;  动态规划的基本思想很简单，在求解某个问题时，我们需要解其不同部分（即子问题），将其记录在一个表中，根据子问题的解以得出原问题的解（**状态转移矩阵**）

 &emsp; &emsp;  动态规划往往用于**优化递归问题**，例如斐波那契数列，如果运用递归的方式来求解会重复计算很多相同的子问题，利用动态规划的思想可以减少计算量。

 &emsp; &emsp;  **动态规划是自底向上，递归树是自顶向下**。

 &emsp; &emsp; 与分治法不同，适用于动**态规划求解的问题经分解得到的子问题往往不是相互独立的**。

 &emsp; &emsp;  通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

<br>

**常用的解题步骤**

1. **确定子问题：** 在这一步重点是分析那些**变量是随着问题规模的变小而变小的**， 那些变量与问题的规模无关。 
2. **确定状态：**根据上面找到的子问题来给你分割的**子问题限定状态** 。
3. **推到出状态转移方程：**这里要注意你的状态转移方程是不是满足所有的条件， 注意不要遗漏。 
4. **确定边界条件（开始点）**：根据题目的信息来找到动态规划的开始点
5. **确定实现方式：**依照个人习惯，就像是01背包的两层for循环的顺序 。
6. **确定优化方法：**很多时候你会发现走到这里的时候你需要返回第1步重来。首先考虑降维问题（优化内存），优先队列、四边形不等式（优化时间）等等

<br>

## 基础DP

 &emsp;  &emsp; 这类DP主要是一些状态比较容易表示，转移方程比较好想，问题比较基本常见的。主要包括递推、背包、最长递增序列（LIS），最长公共子序列（LCS）

### 1）递推问题

PS：递推与递归是不同的

> **递归**：直接或间接的自身调用自身，它通常把一个大型复杂的问题层层转化为一个与原问题相似的规模较小的问题来求解，先将现在要求的大问题放在原处不作处理，转而将其转化成几个小问题，在解决完小问题的基础上再回到原处来解决这个大问题，因此递归策略只需少量的程序就可描述出解题过程所需要的多次重复计算，大大地减少了程序的代码量，但是时间复杂度却比较高，易超时。
>
> **递推**：由前推后，用若干步可重复的简运算或者规律来推下一步。它是按照一定的规律来计算序列中的每个项，通常是通过计算机前面的一些项来得出序列中的指定象的值。
>
> **迭代**：为了逼近所需目标或结果而重复反馈过程的活动。每一次对过程的重复称为一次“迭代”，而每一次迭代得到的结果会作为下一次迭代的初始值。



#### A.斐波那契数列问题

**问题**

 &emsp;  &emsp; 斐波那契数列，又称黄金分割数列，指的是这样一个数列：`1、1、2、3、5、8、13、21、……`。求解第斐波那契数列的第n个值。

**状态转移方程**
$$
F(n)=F(n-1)+F(n-2) \\
F(0)=1 \ \ \ \ \ 
F(1)=1
$$

```python
def Fibonacci(n:int)->int:
    if(n==0):
        return 0
    if(n==1 or n==2):
        return 1
    array=[0 for i in range(n+1)]
    array[1]=1			#表示F(1)
    array[2]=1			#表示F(2)
    for i in range(3,n+1):
        array[i]=array[i-1]+array[i-2]		#依据状态转移方程，进行递推
    return array[n]
```

<br>

#### B.爬楼梯问题

**问题**

 &emsp;  &emsp; 假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**状态转移方程**

 &emsp;  &emsp; 爬到第n阶的方案有两种，从`n-1`阶爬一个台阶，从`n-2`阶爬两个台阶，于是得到状态转移方程
$$
F(n)=F(n-1)+F(n-2)\\
F(1)=1\ \ \ F(2)=2
$$

```python
def climb(n:int)->int:
    if(n==0):
        return 0
    if(n==1):
        return 1
    array=[0 for i in range(n+1)]
    array[1]=1			#表示F(1)
    array[2]=2			#表示F(2)
    for i in range(3,n+1):				
        array[i]=array[i-1]+array[i-2]			#根据状态转移方程得到递推公式
    return array[n]
```

<br>

#### C.最少硬币问题

**问题**

 &emsp;  &emsp; 你有面值为1块、2块、5块的硬币，用尽可能少的硬币凑n块钱，给出最少的硬币的数目

**状态转移方程**

 &emsp;  &emsp; 凑出n块钱的方案有三种：凑出`n-1`块+1块，凑出`n-2`块+2块，凑出`n-5`块加5块。使用最少的那个解决方案
$$
F(n)=
\begin{cases}
min(F(n-1),F(n-2),F(n-5))+1 & & if&n>5\\
min(F(n-1),F(n-2))+1 & & if&n<5
\end{cases}
\\
F(1)=1\ \ \ F(2)=1
$$

```python
def coin(n:int)->int:
    if n == 0:
        return 0
   	if(n==1):
        return 1
    array = [0 for i in range(n+1)]
    array[1] = 1		#表示F(1)
    array[2] = 1		#表示F(2)
    for i in range(3, n+1):
        if i >= 5:
            array[i] = min(array[i-1], array[i-2], array[i-5]) + 1
        else:
            array[i] = min(array[i-1], array[i-2]) + 1
    
    return array[n]
```

<br>

<br>

### 2) 背包问题

#### A.0-1背包问题

**问题**

 &emsp;  &emsp; 有一个背包可以存放M斤物品，有N种物品（每件物品只有1件），他们重量分别是w1,w2,w3..........，他们价值分别是p1,p2,p3...............。问怎么装载物品，使背包装的载物品价值最大？

**例子**

 &emsp;  &emsp; 背包装10斤物品，有3件物品，重量分别是3斤，4斤，5斤，价值分别是4，5，6；问怎么装载物品，使背包装的载物品价值最大？

**思路**

 &emsp;  &emsp; 我们可以构造一个数据结构C，其中C是一个`N+1`行，`M+1`列的矩阵。第`i+1`行表示，当只能存放前i个物品的情况；第j列表示，当前剩余容量为j斤。`c[i][j]` 表示当只能存放前i个物品，且剩余容量为j斤时的最大价值。由该数据结构，我们可以知道，第`N+1`行的情况，就是本次我们要解决的问题。`C[N][M]`就是我们要的结果

 &emsp;  &emsp; 对于本题，数据结构显示如下：

![1](https://github.com/BaiDingHub/Blog_images/blob/master/%E7%AE%97%E6%B3%95/%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%E4%B9%8B%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%EF%BC%88%E4%B8%80%EF%BC%89%E5%9F%BA%E7%A1%80DP/1.png?raw=true)

 &emsp;  &emsp; 我们自顶向下进行考虑，我们根据是否选取第`N`个物品进行判断，当总容量够放第`N`件物品时，则总价值等于第N件物品的价值+剩下`N-1`件在剩余容量下的总价值。当总容量不够放第N件物品时，总价值等于剩下`N-1`件物品在总容量下的总价值。

 &emsp;  &emsp; 其**状态转移方程**为：
$$
C[N][M]=
\begin{cases}
max(p[N]+C[N-1][M-W[N]],C[N-1][M]) & & if&M>W[N]\\ 
C[N-1][M] & & if&M<W[N]
\end{cases}
$$
 &emsp;  &emsp; 根据状态转移方程，我们可以自底向上的写代码。

```python
def zero_one_bag(W:List[int],P:List[int],M:int)->int:		# W表示重量列表，P表示价值列表，M表示允许重量
    N=len(W)
    C=[[0 for i in range(M+1)] for j in range(N+1)]			#构造N+1行，M+1列的数据结构
    for i in range(1,N+1):
        for j in range(1,M+1):
            if j>=W[i-1]:			#如果满足条件，进行更新
            	C[i][j]=max(P[i-1]+C[i-1][j-W[i-1]],C[i-1][j])
            else:
                C[i][j]=C[i-1][j]
    return C[N][M]
```

**优化**

 &emsp;  &emsp; 上面的解决方案，时间复杂度为`O(MN)`，空间复杂度为`O(MN)`。

 &emsp;  &emsp; 仔细观察状态转移方程，你会发现，在每次计算时，我们只用到了上一行的数据，那么我们只需要保存上一行的数据就行，可以将空间复杂度降为`O(2M)`。

 &emsp;  &emsp; 再仔细观察上图与状态转移方程，实际上，当我们计算C[i][j\]时，我们只用到了上一行的第j列左边的那部分数据，因此我们可以将空间复杂度降为`O(M)`。代码如下：

```python
def zero_one_bag(W,P,M):
    N=len(W)
    C=[0 for i in range(M+1)]				#只使用1维列表
    for i in range(N):
        for j in range(M,W[i]-1,-1):   #从M遍历到W(i)，逆序保证了后面的数据的正确更新
            C[j]=max(P[i]+C[j-W[i]],C[j])
    return C[M]
```

<br>

#### B.完全背包问题

 &emsp;  &emsp; 完全背包问题与01背包问题的区别在于每一件物品的数量都有无限个，而01背包每件物品数量只有一个。

**问题**

 &emsp;  &emsp; 有一个背包可以存放M斤物品，有N种物品（每件物品有无限件），他们重量分别是w1,w2,w3..........，他们价值分别是p1,p2,p3...............。问怎么装载物品，使背包装的载物品价值最大？

**简单思路**

 &emsp;  &emsp; 将其转换成0-1背包，对每件物品增加一个循环，其状态转移方程为：
$$
C[N][M]=max(k*p[N]+C[N-1][M-k*W[N]],C[N-1][M],C[M])\ \ \ \ \  0\le k*W[N]\le M
\\
在这里k=0，为上面的第二种情况
$$
**优化**

 &emsp;  &emsp; 上面的解决方案想法简单，但增加了一个循环，时间复杂度为O(NMK)，我们可以利用0-1背包问题中的优化思路的解决方案，只需要将原来的逆序改成顺序。

 &emsp;  &emsp; 为什么将原来的逆序改成顺序，就从一个物体转换成无限个物体了呢？因为，在我们更新后面的值的时候，我们使用的是已经更新过的值，即当C[j]=P[i]的时候，就已经对添加一个物体做出了测试，当`C[j]=2*P[i]`时，使用的是`C[j]=P[i]`更新过的值，即此时，是对添加两个物体做出了测试，····。我们就可以得到我们最后的结果。

```python
def exclusive_bag(W:List[int],P:List[int],M:int)->int:
    N=len(W)	
    C=[0 for i in range(M+1)]				#只使用1维列表
    for i in range(N):
        for j in range(W(i),M+1):   #从W(i)遍历到M，顺序保证了可以添加无限多个物体
            C[j]=max(P[i]+C[j-W[i]],C[j])
    return C[M]
```

<br>

#### C.多重背包问题

 &emsp;  &emsp; 多重背包问题与01背包问题、完全背包问题的区别在于每一件物品的数量都是固定的

**问题**

 &emsp;  &emsp; 有一个背包可以存放M斤物品，有N种物品，他们的数量分别时v1,v2,v3..........，他们重量分别是w1,w2,w3..........，他们价值分别是p1,p2,p3...............。问怎么装载物品，使背包装的载物品价值最大？

**思路**

 &emsp;  &emsp; 将其转换成0-1背包，对每件物品增加一个循环，其状态转移方程为：
$$
C[N][M]=max(k*p[N]+C[N-1][M-k*W[N]],C[N-1][M],C[M])\ \ \ \ \  0\le k\le V[N]
\\
在这里k=0，为上面的第二种情况,注意临界条件，这里就不写了
$$

```python
def multi_bag(W:List[int],P:List[int],V:List[int],M:int)->int:
    N=len(W)
    C=[0 for i in range(M+1)]				#只使用1维列表
    for i in range(N):
        for j in range(M,W(i)-1,-1):   #从M遍历到W(i)，逆序保证了后面的数据的正确更新
            for k in range(V[i]):
                if(k*W[i]<=j):
            		C[j]=max(k*P[i]+C[j-k*W[i]],C[j])
    return C[M]
```

<br>

<br>

### 3）序列问题

#### A.最长递增序列（LIS）

[leetcode--题300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

**问题**

 &emsp;  &emsp; 给定一个无序的整数数组，找到其中最长上升子序列的长度。一个字符串的 *子序列* 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

**例子**

 &emsp;  &emsp; [4, 2, 3, 1, 5]的最长递增子序列为[2 3 5]，长度为 3 。

**简单思路**

 &emsp;  &emsp; 记dp[i]表示当算到第i个元素时的最长递增子序列长度。易知，dp[0]=1。当计算dp[i]时，要将arr[i]与前面i个元素进行一一对比，如果小于，则等于其dp值+1，可得如下状态转移方程：
$$
dp[i]=max(dp[k_1],dp[k_2],...)+1 \ \ \ \ \ \ \ \ \ when\ \ arr[k_1],arr[k_2],...<arr[i]
\\
dp[0]=1
$$

 &emsp;  &emsp; 算法时间复杂度O(n^2)

```python
def lengthOfLIS(arr:List[int])->int:
    if not nums:
        return 0
    N=len(arr)
    dp=[0 for i in range(N)]
    dp[0]=1
    for i in range(1,N):
        dp[i]=1
        for k in range(i):
            if(arr[k]<arr[i] and dp[k]+1>dp[i]):	#与前i个元素的值一一对比，如果满足条件
                dp[i]=dp[k]+1
    return max(dp)			#返回最大值
```

**优化**  贪心+二分

 &emsp;  &emsp; 我们可以将时间复杂度优化到`O(n logn)`，看到`log n`，我们就知道该优化方法大概率要用二分了。

 &emsp;  &emsp; 我们观察上面的动态规划求解过程，我们容易发现，在对第i个元素进行讨论时，我们只使用到了前面元素中比 第i个元素小的 元素的 最长递增子序列的长度。因此，我们可以建立一个**数据结构result**，result记录了当前**最长递增子序列的长度**，以及在该长度下，**最小的末尾元素**（注意，该result记录的内容并不是最长递增子序列）。我们会将列表中的元素逐步加入到result中，让我们看看如何去维护该数据结构。

- 将4加入`result`，result的元素为`[4]`，这表示，在输入为[4]时最长递增子序列长度最大为1，且存在某个最长递增子序列最小末尾元素为4
- 将2加入`result`，由于2<4。所以此时，最长递增子序列的长度最大为1，且存在某个最长递增子序列的最小末尾元素为2。此时`result`将会变成`[2]`
- 将3加入`result`，由于3>2。所以此时，原本长度为1的最长递增子序列将会扩增，那个序列长度将会变成2，且该序列的最小末尾元素为3，此时`result`将会变成`[2，3]`
- 将1加入`result`，`1<2`。而我们并不需要`result`中不是末尾的元素，即2的存在不重要。我们可以将其替换成1，以来保证在将来会出现以1为开始的最长递增子序列。此时`result`将会变成`[1，3]`
- 将5加入`result`，由于`5>3`。所以此时，原本长度为2的最长递增子序列将会扩增，那个序列长度将会变成3，且该序列的最小末尾元素为5，此时`result`将会变成`[1，3，5]`。此时result更新完成

 &emsp;  &emsp; 我们对上面的过程进行总结，即遍历列表，逐步往`result`添加元素，如果该元素大于`result`内的所有值，则将其添加到`result`的末尾；如果该元素并不是大于`result`内的所有值，则替换掉`result`内比该元素大的最小的那个值。

 &emsp;  &emsp; 最后的`result`的长度就是我们的结果，我们可以对上面的插入过程进行优化，使用二分插入的方法，将插入时间缩减到`O(log n)`。

```python
def lengthOfLIS(nums: List[int]) -> int:
    result = []
    for num in nums:
        if not result or num>result[-1]:	#如果该元素大于result内的所有值
            result.append(num)
        else:
            l, r= 0, len(result)-1		#二分法的左右指针
            loc = r						#标记result中大于num的最小值的index
            while(l<=r):				#二分查找找到大于num的最小值
                mid = (l+r)//2
                if(result[mid]>=num):
                    loc = mid
                    r = mid-1
                else:
                    l = mid+1
            result[loc]=num
    return len(result)
```

<br>

#### B.最长公共子序列（LCS）

[leecode--题1143](https://leetcode-cn.com/problems/longest-common-subsequence/)

**问题**

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长公共子序列。

**举例**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

**思路**

![2](https://github.com/BaiDingHub/Blog_images/blob/master/%E7%AE%97%E6%B3%95/%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%E4%B9%8B%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%EF%BC%88%E4%B8%80%EF%BC%89%E5%9F%BA%E7%A1%80DP/2.jpg?raw=true)

- `dp[i][j]`表示`text1[0:i]`与`text2[0:j]`的最长公共子序列的长度
- 当`text1[i-1]==text2[j-1]`说明这两个字符是公共的字符，只要考察其子问题，`dp[i-1][j-1]`的长度即可，在此基础上+1,
- 当`text1[i-1]!=text2[j-1]`,说明这两个字符不是公共的字符，只要考察其两个子问题，`dp[i-1][j],dp[i][j-1]`,取max
- 注意这里用的是`text1[i-1]`与`text2[j-1]`。我们增加了一行与一列空串，使我们能够方便的初始化`text1[0]`那一行与`text2[0]`那一列的值

**状态转移方程**
$$
dp[i][j]=
\begin{cases}
dp[i-1][j-1]+1 & & if&text1[i-1]==text2[j-1]\\ 
max(dp[i][j-1],dp[i-1][j]) & & if&text1[i-1]!=text2[j-1]
\end{cases}
$$

```python
def lengthOfLCS(text1: str, text2: str) -> int:
    N=len(text1)
    M=len(text2)
    if N==0 or M==0:
        return 0
    dp=[[0 for i in range(M+1)] for j in range(N+1)]		#构造N行M列的数据结构
    for i in range(1,N+1):
        for j in range(1,M+1):
            if text1[i-1]==text2[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i][j-1],dp[i-1][j])
    return dp[N][M]
```

**优化**

这个题可以像背包问题那样，将矩阵转换成两个一维数组来进行计算。

<br>

#### C.最长连续序列和

[leetcode--题53最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

**问题**

 &emsp;&emsp;  给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例**

```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**思路**

 &emsp;&emsp;  此次思路采用动态规划与双指针算法思想得融合。首先初始化左右指针指向首元素，右指针开始移动，计算左指针到右指针得总和，若小于0，则左指针移动到右指针得位置，右指针继续移动，若和大于最大值，则更新最大值。

```python
def maxSubSequence(arr:List[int])->int:
    thisSum = maxSum = 0
    for a in arr:
        thisSum += a
        if thisSum > maxSum:
            maxSum = thisSum
        elif thisSum <0:
            thisSum = 0
    return maxSum
```

<br>

<br>

### 4) 正则表达式匹配问题

#### A.正则表达式匹配

[leetcode--题10正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

**问题**

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

```
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
```

说明:

```
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
```

**例子**

```
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**思路**

- 记`N=len(text),M=len(pattern)`，建立一个`N+1`行，`M+1`列的`dp`bool矩阵，其中`dp[i][j]`表示`text[i:]` 与`pattern[j:]` 是否匹配。
- 初始化所有值为`False`，我们很容易就知道`dp[-1][-1]=True`，因为两个空值，一定匹配。于是我们从后往前进行遍历，直到计算出`dp[0][0]`
- 当遍历到`dp[i][j]`时，其可能碰到的情况如下：
  - 定义`first_match`表示`text[i]`与`pattern[j]`是否匹配，因为正则表达式`.`的存在，我们可以得到`first_match = pattern[j] in {text[i],'.'}`
  - 1、如果`pattern[j+1]` 为正则表达式`*`时，我们要考虑两种情况
    - 第一种：该`*`表示零个`pattern[j]`的情况，此时`dp[i][j] = dp[i][j+2]`
    - 第二种：该`*` 表示多个`pattern[j]`的情况，此时如果`first_match`为`True`，则我们要考虑`text[i+1]`是否与`pattern[j]`相同（即`dp[i+1][j]`是否为`True`)
  - 综合上面的情况，我们可以得到此时`dp[i][j] =dp[i][j+2] or(first_match and dp[i+1][j])`
  - 2、如果`pattern[j+1]` 不是正则表达式`*`时，我们只需要考虑`text[i]与pattern[j]`是否匹配，以及之后的字符串是否匹配，即`dp[i][j]=first_match and dp[i+1][j+1]`



**状态转移方程**
$$
first\_match = pattern[j]\quad in\quad \{text[i]，'.'\}
\\
dp[i][j]=
\begin{cases}
dp[i][j+2]\quad or\quad (first\_match\quad and\quad dp[i+1][j]) & & pattern[j+1]=='*' \\ 
first\_match\quad and\quad dp[i+1][j+1] & & pattern[j+1] != '*' 
\end{cases}
$$


```python
def isMatch(text:List[int], pattern:List[int])->int:
    #初始化
    dp = [[False] * (len(pattern)+1) for _ in range(len(text)+1)]
    dp[-1][-1] = True
    
    #逆序进行遍历dp矩阵
    #我们需要考虑第len(text)行，是为了保证""与"a*"情况的考虑
    for i in range(len(text),-1,-1):
        for j in range(len(pattern)-1,-1,-1):
            first_match = i < len(text) and pattern[j] in {text[i],'.'}
            #判断下一个字符是否为'*'
            if j+1 < len(pattern) and pattern[j+1] == '*':
                dp[i][j] = dp[i][j+2] or (first_match and dp[i+1][j])
            else:
                dp[i][j] = first_match and dp[i+1][j+1]
    return dp[0][0]
```





**参考链接**

- [leetcode动态规划](https://leetcode-cn.com/tag/dynamic-programming/) 
- [动态规划总结与题目分类](https://blog.csdn.net/eagle_or_snail/article/details/50987044)
- [动态规划入门之dp递推～](https://blog.csdn.net/lxt_Lucia/article/details/81100723)
- [0-1背包问题动态规划详解](https://www.cnblogs.com/usa007lhy/archive/2013/05/19/3087195.html)
- [最长递增子序列（LIS）](https://blog.csdn.net/qq_41765114/article/details/88415541?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
- [阿飞学算法-1143. 最长公共子序列(一维数组，压缩空间，多解法)](https://leetcode-cn.com/problems/longest-common-subsequence/solution/a-fei-xue-suan-fa-zhi-by-a-fei-8/)

