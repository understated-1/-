# PyTorch文档阅读

## Torch文档阅读

简述：包torch包含了多维张量的数据结构以及基于其上的多种数学操作。

### 张量Tensors

```python
import torch
# 确定obj对象是否为torch张量
torch.is_tensor(obj)   
# 确定obj对象是否为pytorch storage对象
torch.is_storage(obj)
# 返回input张量中元素个数
torch.numel(input) -> int
```

#### 例子

```python
# torch.numel实例
import torch
test_data = torch.ones(3 , 3)
print(torch.numel(test_data))  #标准输出9
```

### 创建操作Creation Ops

```python
# 创建对角线位置全为1，其余位置均为0的2维张量
torch.eye(n, m=None, out=None)
# 从numpy数组转化得到tensor
# 注意numpy数组与torch.Tensor会共享同一内存空间
# 且该张量不能改变大小
torch.from_numpy(ndarray) -> Tensor
# (伪)(等差)数列函数
# torch.linspace定义start, end区间内steps个点
# torch.logspace定义10^start, 10^end区间内stpes个点
# torch.arange定义start,end区间内以step为间隔的点
torch.linspace(start, end, stpes=100, out=None)
torch.logspace(start, end, steps=100, out=None)
torch.arange(start, end, steps=1, out=None)
# torch.ones返回全为1的张量，形状由size定义
# size为整数序列
torch.ones(*size, out=None)
# torch.rand返回在区间【0，1）上均匀分布中抽取的一组随机数
# 形状由size定义
torch.rand(*size, out=None)
# torch.randn类似上述，但是在标准正态分布机制下抽取随机数
torch.randn(*size, out=None)
# torch.randperm返回由0至n-1的随机整数排列
torch.randperm(n, out=None)
# torch.zeros返回全为标量0的张量
torch.zeros(*size, out=None)
```

#### 例子

```python
# torch.eye函数实例
import torch
eye_data = torch.eye(5, 4)
```

```python
# torch.from_numpy实例
import numpy as np
import torch
original_data = np.array([1, 2, 3, 4, 5])
converted_data = torch.from_numpy(original_data)
```

```python
# (伪)(等差)数列函数
import torch
data1 = torch.linspace(1, 4, 2)
data2 = torch.logspace(1, 4, 2)
data3 = torch.arange(1, 4, 2)
print(data1)
# tensor([1., 4.])
print(data2)
# tensor([   10., 10000.])
print(data3)
# tensor([1, 3])
```

```python
# torch.randperm实例
import torch
print(torch.randperm(5))
# 随机选取一个输出
# tensor([2, 0, 1, 3, 4])
```

### 索引，切片，连接，换位

```python
# torch.cat在对应维度上对输入的张量序列seq进行连接操作
torch.cat(inputs, dimension=0) -> Tensor
# torch.chunk在给定维度上对输入张量进行分块操作
torch.chunk(tensor, chunks, dim=0)
# torch.gather在给定轴dim，将输入索引张量index指定位置进行聚合
torch.gather(input, dim, index, out=None)
# torch.index_select沿着指定维度进行切片
# 取index中指定的相应项
torch.index_select(input, dim, index, out=None)
# torch.nonzero返回非0元素索引的张量
torch.nonzero(input, out=None)
# torch.split将输入张量分割为相等形状的chunks
# 如果不能整分，则最后一个分块小于其他分块
torch.split(tensor, split_size, dim=0)
# torch.squeeze与torch.unsqueeze
# 进行挤压(将维度1取消)与增维(在指定位置增加维度1)
torch.squeeze(input, dim=None, out=None)
torch.unsqueeze(input, dim, out=None)
# torch.stack沿着新维度对输入张量序列进行连接
torch.stack(sequencem , dim=0)
# torch.t输入2维张量，转置0，1维
torch.t(input, out=None)
# torch.transpose交换维度dim0和dim1
torch.transpose(input, dim0, dim1, out=None)
# torch.unbind删除指定维，返回一个元组
# 包含了沿着指定维切片后的各个切片
```

#### 例子

```python
import torch
x = torch.randn(2, 3)
cat_x_1 = torch.cat((x, x, x), 0)
cat_x_2 = torch.cat((x, x, x), 1)
print(x)
# tensor([[ 1.8385, -0.1973, -0.9161],
#        [-2.8999, -0.5308, -0.6913]])
print(cat_x_1)
# tensor([[ 1.8385, -0.1973, -0.9161],
#        [-2.8999, -0.5308, -0.6913],
#        [ 1.8385, -0.1973, -0.9161],
#        [-2.8999, -0.5308, -0.6913],
#        [ 1.8385, -0.1973, -0.9161],
#        [-2.8999, -0.5308, -0.6913]])
print(cat_x_2)
# tensor([[ 1.8385, -0.1973, -0.9161,  1.8385, -0.1973, -0.9161,  1.8385, -0.1973,
#         -0.9161],
#        [-2.8999, -0.5308, -0.6913, -2.8999, -0.5308, -0.6913, -2.8999, -0.5308,
#         -0.6913]])
```

```python
# torch.chunk实例
import torch
x = torch.ones(2, 3)
[h1, h2] = torch.chunk(x, 2)
[v1, v2, v3] = torch.chunk(x, 3, dim=1)
```

```python
# torch.gather实例
# import torch
t = torch.Tensor([[1,2],[3,4]])
torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
# 1  1
# 4  3
# [torch.FloatTensor of size 2x2]
# 对于3维张量
# out[i][j][k] = tensor[index[i][j][k]][j][k]  # dim=0
# out[i][j][k] = tensor[i][index[i][j][k]][k]  # dim=1
# out[i][j][k] = tensor[i][j][index[i][j][k]]  # dim=3
# 需要保证index与input的形状相同
```

```python
# torch.index_select实例
>>> x = torch.randn(3, 4)
>>> x

 1.2045  2.4084  0.4001  1.1372
 0.5596  1.5677  0.6219 -0.7954
 1.3635 -1.2313 -0.5414 -1.8478
[torch.FloatTensor of size 3x4]

>>> indices = torch.LongTensor([0, 2])
>>> torch.index_select(x, 0, indices)

 1.2045  2.4084  0.4001  1.1372
 1.3635 -1.2313 -0.5414 -1.8478
[torch.FloatTensor of size 2x4]

>>> torch.index_select(x, 1, indices)

 1.2045  0.4001
 0.5596  0.6219
 1.3635 -0.5414
[torch.FloatTensor of size 3x2]
```

```python
# torch.mask_select实例
# torch.nonzero实例
>>> x = torch.randn(3, 4)
>>> torch.masked_select(x, x > 0)
tensor([0.2336, 0.8238, 0.0586, 1.4704, 0.5959, 0.4751, 0.8769])
>>> x > 0
tensor([[False,  True,  True,  True],
        [ True, False, False,  True],
        [False,  True, False,  True]])
>>> torch.nonzero(x)
tensor([[0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3]])
```

```python
# torch.split实例
>>> x
tensor([[-0.5120,  0.2336,  0.8238,  0.0586],
        [ 1.4704, -0.1781, -1.7767,  0.5959],
        [-1.2976,  0.4751, -1.2451,  0.8769]])
>>> torch.split(x, 2, dim=0)
(tensor([[-0.5120,  0.2336,  0.8238,  0.0586],
        [ 1.4704, -0.1781, -1.7767,  0.5959]]), tensor([[-1.2976,  0.4751, -1.2451,  0.8769]]))
>>> torch.split(x, 2, dim=1)
(tensor([[-0.5120,  0.2336],
        [ 1.4704, -0.1781],
        [-1.2976,  0.4751]]), tensor([[ 0.8238,  0.0586],
        [-1.7767,  0.5959],
        [-1.2451,  0.8769]]))
```

```python
# torch.stack实例
>>> import torch
>>> x = torch.randn(2, 3)
>>> cat_x_1 = torch.cat((x, x, x), 0)
>>> stack_x_1 = torch.stack((x,x,x), 0)
>>> cat_x_1
tensor([[-0.1513, -0.0334, -0.9547],
        [ 1.5935, -2.3545,  1.3137],
        [-0.1513, -0.0334, -0.9547],
        [ 1.5935, -2.3545,  1.3137],
        [-0.1513, -0.0334, -0.9547],
        [ 1.5935, -2.3545,  1.3137]])
>>> stack_x_1
tensor([[[-0.1513, -0.0334, -0.9547],
         [ 1.5935, -2.3545,  1.3137]],

        [[-0.1513, -0.0334, -0.9547],
         [ 1.5935, -2.3545,  1.3137]],

        [[-0.1513, -0.0334, -0.9547],
         [ 1.5935, -2.3545,  1.3137]]])
>>> cat_x_2 = torch.cat((x, x, x), 1)
>>> cat_x_2
tensor([[-0.1513, -0.0334, -0.9547, -0.1513, -0.0334, -0.9547, -0.1513, -0.0334,
         -0.9547],
        [ 1.5935, -2.3545,  1.3137,  1.5935, -2.3545,  1.3137,  1.5935, -2.3545,
          1.3137]])
>>> 
```

### 随机抽样

```python
# torch.manual_seed设定生成随机数的种子
torch.manual_seed(seed)
# torch.bernouli从伯努利分布中抽取2元随机数(0或者1)
torch.bernoulli(input, out=None)
# torch.multinomial多项分布随机采样
torch.multinomial(input, num_samples,replacement=False, out=None)
```

#### 例子

```python
# torch.bernoulli实例
>>> a = torch.Tensor(3, 3).uniform_(0, 1) # generate a uniform random matrix with range [0, 1]
>>> a

 0.7544  0.8140  0.9842
 0.5282  0.0595  0.6445
 0.1925  0.9553  0.9732
[torch.FloatTensor of size 3x3]

>>> torch.bernoulli(a)

 1  1  1
 0  0  1
 0  1  1
[torch.FloatTensor of size 3x3]

>>> a = torch.ones(3, 3) # probability of drawing "1" is 1
>>> torch.bernoulli(a)

 1  1  1
 1  1  1
 1  1  1
[torch.FloatTensor of size 3x3]

>>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
>>> torch.bernoulli(a)

 0  0  0
 0  0  0
 0  0  0
[torch.FloatTensor of size 3x3]
```

```python
# torch.multinomial实例
>>> weights = torch.Tensor([1, 1, 1, 1, 1, 1])
>>> torch.multinomial(weights, 6, replacement=True)
tensor([2, 5, 2, 4, 5, 1])
>>> torch.multinomial(weights, 6, replacement=False)
tensor([0, 5, 2, 4, 1, 3])
```

###  序列化

```python
# 保存与加载操作
torch.save(obj, f)
torch.load(f)
```

#### 例子

```python
import torch
x = torch.Tensor([1,2,3])
torch.save(x, 'test.txt')
x2 = torch.load('test.txt')
```

### 数学操作

```python
# 计算绝对值
torch.abs(input, out=None)
# 计算反余弦
torch.acos(input, out=None)
# 对于输入张量input逐元素加入标量值value
torch.add(input, value, out=None)
# 计算过程
# 用tensor2对tensor1逐元素相除，然后乘以标量值value并加到tensor
torch.addcdiv(tensor, value=1, tensor1, tensor2, out=None)
# 用tensor2对tensor1逐元素相乘，并对结果乘以标量值value然后加到tensor
torch.addcmul(tensor, value=1, tensor1, tensor2, out=None)
# 计算反正弦
torch.asin(input, out=None)
# 计算反正切
torch.atan(input, out=None)
# 向上取整
torch.ceil(input, out=None)
# 夹紧到区间[min, max]
# 小于min的值取为min
# 大于max的值取为max
torch.clamp(input, min, max, out=None)
# 计算余弦
torch.cos(input, out=None)
# 计算双曲余弦
torch.cosh(input, out=None)
# 标量除法
torch.div(input, value, out=None)
# 指数
torch.exp(tensor, out=None)
# 向下取整
torch.floor(tensor, out=None)
# 计算除法余数，余数正负与被除数相同
torch.fmod(input, divisor, out=None)
# 返回分数部分
torch.frac(tensor, out=None)
# torch.lerp线性插值
# 即out_i = start_i + weight * (end_i - start_i)
torch.lerp(start, end, weight, out=None)
# 计算对数
torch.log(input, out=None)
# 计算input+1的对数
torch.log1p(input, out=None)
# 标量乘法
torch.mul(input, value, out=None)
# 取负运算
torch.neg(input, out=None)
# 取幂值运算
# exponent可为float或者张量
torch.pow(input, exponent, out=None)
# 倒数计算
torch.reciprocal(input, out=None)
# 除法余数
torch.remainder(input, divisor, out=None)
# 四舍五入计算
torch.round(input, out=None)
# 平方根倒数计算
torch.rsqrt(input, out=None)
# sigmoid值计算
torch.sigimod(input, out=None)
# 正负返回
torch.sign(input, out=None)
# 正弦计算
torch.sin(input, out=None)
# 双曲正弦计算
torch.sinh(input, out=None)
# 平方根计算
torch.sqrt(input, out=None)
# 正切计算
torch.tan(input, out=None)
# 双曲正切计算
torch.tanh(input, out=None)
# 舍弃小数操作
torch.trunc(input, out=None)
# 指定维度累计积
torch.cumprod(input, dim , out=None)
# 指定维度累计和
torch.cumsum(input, dim, out=None)
# 计算input-other的p范数
torch.dist(input, other, p=2, out=None)
# 计算均值
torch.mean(input, dim, out=None)
# 返回输入张量在给定维度每行的中位数
# 同时返回一个包含中位数索引的LongTensor
torch.median(input, dim=1, value=None, indices=None) -> (Tensor, LongTensor)
# 返回众数，类似上述
torch.mode(input, dim=1, values=None, indices=None) -> (Tensor, LongTensor)
# 返回p范数
torch.norm(input, p = 2)
# 返回输入张量input所有元素的乘积
torch.prod(input)
# 返回输入张量input所有元素的标准差
torch.std(input)
# 返回输入张量input所有元素的和
torch.sum(input)
# 返回输入元素所有元素的方差
torch.var(input)
# 比较操作
torch.eq(input, other, out=None) # other可为数可为值
torchequal(tensor1, tensor2) #相同形状和元素值
torch.ge(input, other, out=None) # 大于等于
torch.gt(inout, other, out=None) #大于
# 取input指定维上第k个最小值， 若不指定，默认为input的最后一维
torch.kthvalue(input, k , dim=None, out=None)
torch.le(input, other, out=None)
torch.lt(input, other, out=None)
torch.max()
torch.max(input, dim, max=None, max_indices=None)
torch.min()
torch.min(input, dim, min=None, min_indices=None)
torch.ne(input, other, out=None) # not equal
# 排序操作（默认升序）
torch.sort(input, dim=None, descending=False, out=None)
# 取dim维度输入张量input中k个最大值
# 若lagest为假，返回最小的k个值
torch.topk(input, k, dim=None, largest=True, out=None)
# 叉积
torch.cross(input, other, dim=-1, out=None)
# 对角线
torch.diag(input, diagonal=0, out=None)
# 计算输入张量的直方图
torch.histc(input, bins=100, min=0, max=0, out=None)
# 规范化张量
torch.renorm(input, p, dim, maxnorm, out=None)
# 返回输入2维矩阵对角线元素的和(迹)
torch.trace(input)
# 返回矩阵的下三角部分
torch.tril(input, k = 0 , out=None)
# 返回矩阵的上三角部分
torch.triu(input, k = 0 , out=None)
# 张量点乘(两个张量均为1维向量)
torch.dot(tensor1, tensor2)
# 返回特征值与特征向量
torch.eig(a, eigrnvector=False, out=None)
# 输入二维矩阵取逆
torch.inverse(input, out=None)
# 矩阵乘法
torch.mm(mat1, mat2, out=NOne)
```

#### 例子

```python
# torch.addcdiv
import torch
t = torch.randn(2,3)
t1 = torch.randn(2,3)
t2 = torch.randn(2,3)
torch.addcdiv(t,0.1,t1,t2)
```

```python
# torch.mul实例
>>> import torch
>>> a = torch.arange(1, 5)
>>> a
tensor([1, 2, 3, 4])
>>> b = torch.arange(2, 6)
>>> c = 2
>>> a
tensor([1, 2, 3, 4])
>>> b
tensor([2, 3, 4, 5])
>>> c
2
>>> torch.mul(a, c)
tensor([2, 4, 6, 8])
>>> torch.mul(a, b)
tensor([ 2,  6, 12, 20])
```

```python
# torch.diag实例
>>> a = torch.arange(0, 4)
>>> a
tensor([0, 1, 2, 3])
>>> torch.diag(a)
tensor([[0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])
>>> b = torch.diag(a)
>>> torch.diag(b)
tensor([0, 1, 2, 3])
```

## torch.Tensor库

### 数据类型

![屏幕截图(842)](C:\Users\DELL\Pictures\Screenshots\屏幕截图(842).png)

torch.Tensor是默认的tensor类型torch.FloatTensor的简称

### Tensor相关方法

#### 创建

```python
torch.FloatTensor([[1,2,3], [4,5,6]])
torch.IntTensot(2, 4,).zero_()
```

#### 通过索引切片获取修改张量tensor中的张量

```python
x = torch.FloatTensor([[1,2,3], [4,5,6]])
print(x[1][2])
x[0][1] = 8
print(x)
```

每一个张量tensor都有一个相应的torch,Storage用来保存其数据

***会改变tensor的函数操作会用一个下划线后缀来表示***

#### 其他操作

```python
# clone操作
clone()
# 复制操作
# 将src中的元素复制到tensor中并返回这个tensor
copy_(src, async=False) 
# 返回单个元素的字节大小
element_size()
# 填充
fill_(value)
# 批量操作，将函数callable作用于tensor中的每一个元素
apply_(callable)
# 返回numpy数组
numpy()
# 调整大小
resize_(*sizes)
# 返回底层内存
storage()
# 0填充
zero_(0)
```

