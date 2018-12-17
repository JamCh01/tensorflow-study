在这里定义了一些tensorflow的基本操作
# 基本操作
## 定义常量
```python
In [2]: a = tensorflow.constant(2)

In [3]: a
Out[3]: <tf.Tensor 'Const:0' shape=() dtype=int32>
```

## 使用默认图进行操作
```python
In [7]: with tensorflow.Session() as session:
            print('a={a}, b={b}'.format(a=2, b=3))
            print("Addition with constants: {}".format(session.run(a+b)))
            print("Multiplication with constants: {}".format(session.run(a*b)))
        a=2, b=3
Addition with constants: 5
Multiplication with constants: 6
```

## 定义操作
```python
In [5]: add = tensorflow.add(a, b)

In [6]: add
Out[6]: <tf.Tensor 'Add:0' shape=() dtype=int32>
```

## 定义占位符
当然了这也是一种常量，不过和constant的区别是需要进行传值。
```python
In [7]: a = tensorflow.placeholder(tensorflow.int16)

In [8]: a
Out[8]: <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=int16>
```

## 使用占位符
```python
In [10]:with tensorflow.Session() as session:
            print("Addition with variables: {}".format(session.run(add, feed_dict={a: 2, b: 3})))
            print("Multiplication with constants: {}".format(session.run(mul, feed_dict={a: 2, b: 3})))
Addition with variables: 5
Multiplication with constants: 6
```

# 函数细节
## constant（常量）
constant是常量节点，用来传入数据，充当计算的起始节点。
### 创建
```python
In [2]: a = tensorflow.constant(2)

In [3]: a
Out[3]: <tf.Tensor 'Const:0' shape=() dtype=int32>
```
### 参数
可以看到这个函数接收三个参数：  
1. value，初始值，比如int，或list  
2. dtype，数据类型，默认为value的类型  
3. shape，数据形状，默认为value的shape
```python
In [15]: a = tensorflow.constant(value=[[1], [2]], dtype=tensorflow.int32, shape=[1, 2])

In [16]: a
Out[16]: <tf.Tensor 'Const_3:0' shape=(1, 2) dtype=int32>

In [17]: with tensorflow.Session() as session:
    ...:     session.run(a)
    ...:
2018-12-17 18:05:21.655880: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

In [18]: with tensorflow.Session() as session:
    ...:     print(session.run(a))
    ...:
    ...:
[[1 2]]
```

## placeholder（占位符）
placeholder是tensorflow的占位符，也是一种常量。数据由调用`tensorflow.Session().run()`时进行传递。
### 创建
```python
In [7]: a = tensorflow.placeholder(tensorflow.int16)

In [8]: a
Out[8]: <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=int16>
```

### 参数
可以看到这个函数接收三个参数：  
1. dtype，数据类型  
2. shape，数据形状，默认为value的shape  
3. name，常量名
```
In [19]: a = tensorflow.placeholder(tensorflow.int16)
In [20]: with tensorflow.Session() as session:
    ...:     print(session.run(a, feed_dict={a:1}))
    ...:
1
```