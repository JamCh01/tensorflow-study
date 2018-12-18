# 第一个hello tensorflow
## Hello Tensorflow
```python
In [1]: import tensorflow

In [2]: hello = tensorflow.constant('hello tenfowflow')

In [3]: with tensorflow.Session() as session:
   ...:     print(session.run(hello))
   ...:
b'hello tenfowflow'
```

## 低阶API简介
这里是第一个TensorFlow程序，在分析之前当然是需要对TensorFlow进行一些介绍。  
在希望使用Tensorflow做一些事情时，一般需要三个步骤：
- 创建Tensor  
- 添加Operation（op），由op输入一个tensor，输出另外一个tensor  
- 执行计算
### 张量（Tensor）
这个名字可以分解为两个单词理解：Tensor（张量） + Flow（数据流），因而TensorFlow的核心数据单位就是**张量**。张量的阶是他的维数，比如：
```python
3. # 零阶张量，标量，结构为[]
[1., 2., 3.] # 一阶张量，矢量，结构为[3]
[[1., 2., 3.], [4., 5., 6.]] # 二阶张量，矩阵，结构为[2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # 三阶张量，结构为[2, 1, 3]
```
TensorFlow可以使用numpy来表示张量的值。  

### 图（Graph）
TensorFlow中有一个图的概念。在添加某些op时，不会立即执行。TensorFlow会等待所有op添加结束后优化这张图，以便决定如何计算。这也就意味着每个op会作为图的节点，而每个张量会作为图的边。
```python
In [1]: import tensorflow as tf

In [2]: a = tf.constant(3.0, dtype=tf.float32)

In [3]: b = tf.constant(4.0)

In [4]: a
Out[4]: <tf.Tensor 'Const:0' shape=() dtype=float32>

In [5]: b
Out[5]: <tf.Tensor 'Const_1:0' shape=() dtype=float32>

In [6]: total = a + b

In [7]: print(a, b, total)
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32) Tensor("add:0", shape=(), dtype=float32)
```
这里并没有出现预期的`3.0`，`4.0`和`7.0`。这些语句只会构建这张图。图中的每条指令都有自己的名称，这个名称不同于Python的变量名，是根据生成张量的指令命名的，后面会负赘索引（`Const:0`，`Const_1:0`和`add:0`）。

### 会话（Session）
要计算张量的结果，需要实例化一个`tensorflow.Session()`对象。它封装了TensorFlow的运行时的状态，并运行TensorFlow的操作（op）。

#### 创建一个Session对象
```python
In [9]: with tf.Session() as session:
   ...:     print(session.run(total))
   ...:
   ...:
7.0
```
在使用`Session().run()`请求输出时，Tensorflow会回溯整张图，并流经所请求输出的节点的对应的输入值的输入节点。因此会计算出结果。
当然也可以传递给`run()`方法多个张量：
```python
In [10]: with tf.Session() as session:
    ...:     print(session.run({'ab': (a, b), 'total':total}))
    ...:
{'ab': (3.0, 4.0), 'total': 7.0}
```

### 占位符（Placeholder）
图也可以接受参数化的外部输入，这就被称为占位符（类似函数参数）。
```python
In [11]: x = tf.placeholder(tf.float32)

In [12]: y = tf.placeholder(tf.float32)

In [13]: z = x + y

In [14]: with tf.Session() as session:
    ...:     print(session.run(z, feed_dict={x: 3, y: 4.5}))
    ...:     print(session.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
    ...:
7.5
[3. 7.]
```
可以使用`run()`的`feed_dict`参数将数据“投喂”至张量中。