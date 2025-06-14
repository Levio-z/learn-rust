>https://www.youtube.com/watch?v=ThjvMReOXYM
# Crust of Rust: async await

![](Pasted%20image%2020250603120856.png)
- 检查未来是否完成了，没有就yield
- 完成了获取结果
![](Pasted%20image%2020250603124302.png)

### select！
![](Pasted%20image%2020250603125304.png)
- 尝试等待多个future中
- 让等待时间发生，然后做一些事情
- 尝试，没有成功，进入睡眠状态，然后事情发生。执行某些操作。
- 主要是IO。
- 读取磁盘时，async允许其他操作执行。
- 大型状态机，处于当前的状态
	- 多个状态
- select允许你执行分支

![](Pasted%20image%2020250603125748.png)
- 可能处于等待1（等待IO）=>其他事件发生机会
- 等待2
![](Pasted%20image%2020250603130125.png)
- 可能处于等待1（等待IO）=>其他事件
	- 如果我不跑，我让比我高的人选择谁跑
	- wait 我不在继续执行，等待值到来


race！无论哪个被执行，都继续执行
![](Pasted%20image%2020250603130514.png)
它只是通过描述代码在什么情况下可以取得进展以及在什么情况下代码可以产生来描述更改或协作调度一堆计算的机制。

### 问题
race和select的区别

运行时就是将future拉入类似循环的机制，不断地积极尝试

linux Epoll
KQ

外部的执行器循环的底层机制不一样
使用运行时

### 如何在实践中运用
.await就是实现了yield的语法糖

tokio 底层 mayo =>各种系统的底层机制支持

如果废弃的分支有副作用会怎么样

使用IO资源的一一步版本，普通的读取，就是等待直到完成，也不会让出cpu

select 有只运行第一个，或伪随机运行

一般使用future的可变借用，使用本身会导致drop


开销，aync不会给代码带来任何开销

select 只是在一个线程上拉取不同的futire，而不是切换线程

### join

等待所有future完成，
![](Pasted%20image%2020250603140424.png)
 - 串行和并行执行

![](Pasted%20image%2020250603142632.png)
- 迭代器中的所有内容
- 即使执行顺序不一样，但是输出排序和输入顺序一样
![](Pasted%20image%2020250603142855.png)
### 多种关联
![](Pasted%20image%2020250603145317.png)
- 并行，你给他一个future，它将会移动到其他执行器中。
- Excutour 可以有多个线程才有并行性，一个线程没有并行性。
### spawn
- 多线程技术
	- 多线程共享技术
		- Arc+Mutex
![](Pasted%20image%2020250603145841.png)

- 通过通道
- 只读内存，
![](Pasted%20image%2020250603145945.png)
等待其他线程的执行结果，将结果传递会调用者的一种

处理错误
- 打印
- 记录到文件
- 事件分发工具
### future是什么？
  ![](Pasted%20image%2020250603152143.png)

- 状态机使用指针是为了不复制副本
### async
- future的大小取决于其中的堆栈变量
![](Pasted%20image%2020250603154028.png)
- trait 定义异步方法需要使用async_trait
	- 因为future=>pin box
### 让两个future共享状态
arc mutex

为什么需要异步的锁？
以为异步的锁获取不到会让其他future执行，而如果一个future获取锁后，wait，切换到其他future，原来的锁没有释放，还继续尝试获取一个锁，那么就会造成问题。

tokio：：spawn是协作调度。


### 问题

### bug
bug不会包含foo，如果foo里面使用了tokio::spawn
- 使用其他trait跟踪
![](Pasted%20image%2020250603161031.png)

### 使用thread而不是feature
任何计算量大的，没有io的场景，不是真正使用io，没必要实现async