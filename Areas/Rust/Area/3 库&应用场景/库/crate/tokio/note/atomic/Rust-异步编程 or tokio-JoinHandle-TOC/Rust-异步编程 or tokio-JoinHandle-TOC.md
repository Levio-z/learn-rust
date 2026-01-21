---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

JoinHandle 是一个泛型类型，它的 type 参数是生成的任务返回的类型。在上面的例子中，类型是 `JoinHandle<()>` ，一个返回 `String` 的 Future 会生成一个类型为 `JoinHandle<String>` `JoinHandle` 。这个 `JoinHandle` 自身是一个 `Future`,可以获取结果。



### Ⅱ. 应用层
### 实际应用
- 如果你只想让任务自行执行，可以丢弃 `JoinHandle` （丢弃 `JoinHandle` 不会影响被创建的任务）。
- **想要获取已生成任务的执行结果，可以 `.await` 获取任务结果，即等待它完成并使用该结果**。这称为_连接_任务（类似于[连接](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html#method.join)线程，连接的 API 也类似）。[Rust-异步编程 or tokio-JoinHandle-wait](Rust-异步编程%20or%20tokio-JoinHandle-wait.md)

### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-tokio-基本使用](../Rust-tokio-基本使用.md)
	- [Rust-Async和Await-基本概念](../../../../../../../1%20基本概念/2%20进阶/2.3%20并发和异步/Async和Await/Rust-Async和Await-基本概念.md)
- 后续卡片：
	- [Rust-异步编程 or tokio-JoinHandle-wait](Rust-异步编程%20or%20tokio-JoinHandle-wait.md)
	- [Rust-异步编程 or tokio-JoinHandle-wait-panic](Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
