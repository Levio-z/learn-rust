---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层
async/await 的设计理念是让程序员编写 _看似_ 普通的同步代码，但由编译器转换为异步代码

- `async` ：关键字可用于函数签名中来将一个同步函数转换为返回 future 的异步函数。
- `awati`：只有这个关键字本身看起来不太有用。然而，在 `async` 函数内部，`await` 关键字可用于获取一个 future 的异步值：
- [零成本抽象](#零成本抽象)：通过使用 `.await` 运算符，我们无需任何闭包或者 `Either` 类型就可以直接获取 future 的值。于是我们就可以像写普通的同步代码一样编写代码，只不过 _这实际上是异步代码_。


### Ⅱ. 实现层



### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：[RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 基本概念
- [Rust-异步编程-运行时-基本概念](../Rust异步编程/Rust-异步编程-运行时-基本概念.md)
- [Rust-异步编程-async-基本概念](Rust-异步编程-async-基本概念.md)
- [Rust-异步编程-await-基本概念](Rust-异步编程-await-基本概念.md)

### future和任务
- [Rust-future-基本概念-TOC](../Future/Rust-future-基本概念-TOC.md)
- [操作系统-异步编程-任务](../../../../../../../../../Zettelkasten/fleeting/操作系统-异步编程-任务.md)

### 基本示例
- [Rust-tokio-基本使用](../../../../3%20库/库/crate/tokio/note/atomic/Rust-tokio-基本使用.md)中的基本例子，一个运行时任务中这些代码都是顺序执行的，await是顺序执行中可能的暂停点（也就是嵌套的其他异步任务），任务可以在这些暂停点暂停，运行时转而开业调用其他运行时任务
### 生成任务
`await` 可以让当前任务在等待 I/O 或其他事件时进入休眠状态。当这种情况发生时，另一个任务就可以运行了，但是这些其他任务是如何产生的呢？使用 `std::thread::spawn` 来创建一个新任务一样。
[Rust-异步编程 or tokio-spawn单独任务运行异步函数](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-spawn单独任务运行异步函数.md)
### 获取任务结果

[Rust-异步编程 or tokio-JoinHandle](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle.md)
[Rust-异步编程 or tokio-JoinHandle-wait](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-wait.md)
[Rust-异步编程 or tokio-JoinHandle-wait-panic](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)
[Rust-异步编程 or tokio-JoinHandle-abort](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-abort.md)



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-异步编程-async-基本概念](Rust-异步编程-async-基本概念.md)
	- [Rust-异步编程-await-基本概念](Rust-异步编程-await-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
