---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层
### 目录
- 不同的并发编程模型，进程、线程和异步任务
- 异步编程中的阻塞问题
- 异步模型基本组成部分
- async和await编程范式
- 有助于构建新的[并发代码抽象和组合](https://rust-lang.github.io/async-book/part-guide/concurrency-primitives.html)模型
- [并发](https://rust-lang.github.io/async-book/part-guide/sync.html)任务之间的同步问题
- 书中有一章专门介绍[异步编程工具](https://rust-lang.github.io/async-book/part-guide/tools.html) 。
- 最后几章涵盖了一些更专业的课题，首先是[异步销毁和清理](https://rust-lang.github.io/async-book/part-guide/dtors.html) （这是一个常见的需求，但由于目前还没有好的内置解决方案，所以这算是一个比较专业的课题）。
- 本指南的接下来两章将详细介绍 [futures](https://rust-lang.github.io/async-book/part-guide/futures.html) 和 [runtimes](https://rust-lang.github.io/async-book/part-guide/runtimes.html) ，这是异步编程的两个基本构建模块
- 最后，我们来介绍[定时器、信号处理](https://rust-lang.github.io/async-book/part-guide/timers-signals.html)和[异步迭代器](https://rust-lang.github.io/async-book/part-guide/streams.html) （也称为流）。异步迭代器用于处理异步事件序列（与使用 future 或异步函数表示的单个异步事件不同）。这部分内容目前仍在积极开发中，可能还存在一些不足之处。

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 内容
- 首先讨论不同的[并发编程](https://rust-lang.github.io/async-book/part-guide/concurrency.html)模型，包括进程、线程和异步任务
- 第一章将介绍 Rust 异步模型的基本组成部分
- [第二章](https://rust-lang.github.io/async-book/part-guide/async-await.html)深入探讨异步编程的细节
	- 介绍 async 和 await 编程范式
- 更多的异步编程概念

### 异步编程的动机
- 异步编程的主要动机，提高 I/O 性能，我们将在[下一章](https://rust-lang.github.io/async-book/part-guide/io.html)详细介绍。
- 我们也会在同一章详细讨论_阻塞问题_ 。阻塞是异步编程中的一个主要风险，它会导致线程被同步等待的操作（通常是 I/O）阻塞，无法继续执行。
- 异步编程的另一个优点是它有助于构建新的[并发代码抽象和组合](https://rust-lang.github.io/async-book/part-guide/concurrency-primitives.html)模型。讨论完这一点后，我们将继续探讨[并发](https://rust-lang.github.io/async-book/part-guide/sync.html)任务之间的同步问题。




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
