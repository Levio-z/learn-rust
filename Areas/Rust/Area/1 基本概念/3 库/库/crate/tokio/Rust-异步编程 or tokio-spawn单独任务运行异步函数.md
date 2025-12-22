---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

如果我们使用 `thread::spawn` 或 `tokio::spawn` 我们就会引入并发性，并且可能引入并行性，前者是在线程之间，后者是在任务之间。

- `tokio::spawn` 是同步函数，它会**立即安排一个任务**在 Tokio 的运行时中执行。**`spawn` 函数本身已经做了调度工作，所以调用 `spawn` 时不需要 `.await` 来“启动任务”**。
- 它返回一个 [JoinHandle](Rust-异步编程%20or%20tokio-JoinHandle.md)


### Ⅱ. 应用层

### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 基本例子
- 前置：[Rust-tokio-基本使用](Rust-tokio-基本使用.md)
```rust
use tokio::{spawn, time::{sleep, Duration}};

async fn say_hello() {
    // Wait for a while before printing to make it a more interesting race.
    sleep(Duration::from_millis(100)).await;
    println!("hello");
}

async fn say_world() {
    sleep(Duration::from_millis(100)).await;
    println!("world!");
}

#[tokio::main]
async fn main() {
    spawn(say_hello());
    spawn(say_world());
    // Wait for a while to give the tasks time to run.
    sleep(Duration::from_millis(1000)).await;
}
```

我们有两个函数分别打印“hello”和“world!”。但这次我们并发（并行）运行它们，而不是顺序运行。如果你运行程序几次，应该会看到字符串以两种顺序打印——有时先打印“hello”，有时先打印“world!”。典型的并发竞争！

[操作系统-异步编程-任务](../../../../../../../../Zettelkasten/fleeting/操作系统-异步编程-任务.md)中的运行时任务概念。
- spawn `spawn` 接收一个 future（请记住，它可以由许多更小的 future 组成），并将其作为一个新的 Tokio 任务运行。Task 是 Tokio 运行时调度和管理的概念（而不是单个 future）。
[Rust-tokio-基本概念](Rust-tokio-基本概念.md)中多线程运行时概念
- 任务可能运行在不同线程上，不同的任务可能并发或并行运行





## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-tokio-基本使用](Rust-tokio-基本使用.md)
	- [Rust-Async和Await-基本概念](../../../../1%20基础知识/RustBook/17.Async和Await/Async和Await/Rust-Async和Await-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
