这里是第一个tensorflow程序，在分析之前当然是需要对tensorflow进行一些介绍。  
这个名字可以分解为两个单词理解：Tensor（张量） + Flow（数据流）  
在希望使用Tensorflow做一些事情时，一般需要三个步骤：  
1. 创建Tensor  
2. 添加Operation（op），由op输入一个tensor，输出另外一个tensor  
3. 执行计算  

Tensorflow中有一个图的概念，op会作为节点添加至图中。在添加某些op时，不会立即执行。Tensorflow会等待所有op添加结束后优化这张图，以便决定如何计算。  

比如上面的代码中的`hello`便是一个`Constant op`。构造函数返回的值表示`Constant op`的输出。  
而`Session`提供了op的计算环境。使用会话管理可以很优雅地调用Session。之后执行这张图即可获得结果。