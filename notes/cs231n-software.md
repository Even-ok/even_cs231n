1. GPU的位置

![image-20230719174021577](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230719174021577.png)

2. cpu vs. gpu

![image-20230719174225587](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230719174225587.png)

- gpu是不可以以一个核单独运行的
- gpu有自己的RAM
- 矩阵乘法适合gpu运行，cpu可能只能按顺序计算。

3. 效率比较

![image-20230719175026868](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230719175026868.png)

 

4. gpu处理速度是很快的，内存什么的也需要在快速的时间内快速地提供数据给gpu，不然会遇到bottleneck。
5.   cpu的多个线程交错从内存中读取数据，给gpu处理，就像gpu也可以并行交错计算一样。但深度学习框架已经帮我们处理好了一部分的东西。
6. 深度学习框架，第一个是Caffe。他们首先来自学术界。

![image-20230719180607175](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230719180607175.png)

7. tensorflow常见模式
   - 首先定义好计算图（上半部分）
   - 定义好了之后喂数据

![image-20230719181615697](C:\Users\12849\AppData\Roaming\Typora\typora-user-images\image-20230719181615697.png)

8. 把图中w1  w2的placeholder换成variable，存储在计算图里面 