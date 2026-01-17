---
tags:
  - permanent
---
## 1. 核心观点  

异步编程基本概念：[操作系统-异步编程-基本概念](操作系统-异步编程-基本概念.md)

## 2. 背景/出处  
- 来源：
	- https://rust-lang.github.io/async-book/part-guide/concurrency.html#fr-proc-program-1
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 异步编程
- [操作系统-异步编程-基本概念](操作系统-异步编程-基本概念.md)
### Rust异步编程基础
#### 基本概念
- [Rust-Async和Await-基本概念](../Async和Await/Rust-Async和Await-基本概念.md)
	- [Rust-异步编程-运行时-基本概念](Rust-异步编程-运行时-基本概念.md)
	- [Rust-异步编程-async-基本概念](../Async和Await/Rust-异步编程-async-基本概念.md)
	- [Rust-异步编程-await-基本概念](../Async和Await/Rust-异步编程-await-基本概念.md)
- [Futures and tasks  未来与任务](https://rust-lang.github.io/async-book/part-guide/async-await.html#futures-and-tasks)
	- [操作系统-异步编程-任务](../../../../../../../../Zettelkasten/fleeting/操作系统-异步编程-任务.md)
	- [Rust-future-基本概念-TOC](../Future/Rust-future-基本概念-TOC.md)
	- [Rust-future-状态机](../Future/Rust-future-状态机.md)

#### 基本示例
- [Rust-tokio-基本使用](../../../../3%20库/库/crate/tokio/note/atomic/Rust-tokio-基本使用.md)
	- 中的基本例子，一个运行时任务中这些代码都是顺序执行的，await是顺序执行中可能的暂停点（也就是嵌套的其他异步任务），任务可以在这些暂停点暂停，运行时转而开业调用其他运行时任务
#### 生成任务
`await` 可以让当前任务在等待 I/O 或其他事件时进入休眠状态。当这种情况发生时，另一个任务就可以运行了，但是这些其他任务是如何产生的呢？使用 `std::thread::spawn` 来创建一个新任务一样。
- [Rust-异步编程 or tokio-spawn单独任务运行异步函数](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-spawn单独任务运行异步函数.md)
#### 获取任务结果
- [Rust-异步编程 or tokio-spawn单独任务运行异步函数](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-spawn单独任务运行异步函数.md)
- [Rust-异步编程 or tokio-JoinHandle](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle.md)
- [Rust-异步编程 or tokio-JoinHandle-wait](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-wait.md)
- [Rust-异步编程 or tokio-JoinHandle-wait-panic](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)
- [Rust-异步编程 or tokio-JoinHandle-abort](../../../../3%20库/库/crate/tokio/note/atomic/Rust-异步编程%20or%20tokio-JoinHandle-abort.md)



### 学习async可能的挑战和收益
- 收益：在典型的并发工作负载下表现出色的运行时**性能**。
- 挑战：
	- More frequent interaction with advanced language features, such as lifetimes and pinning.
	- Some **compatibility constraints**（兼容性限制）, both between sync and async code, and between different async runtimes.
	- 由于异步运行时和语言支持的不断演进，**维护负担也更高**。
- 简而言之，异步 Rust 比同步 Rust 更难使用，维护负担也更高，但能带来同类最佳性能。所有异步 Rust 领域都在不断改进，因此这些问题的影响会随着时间逐渐消退。
### Rust 语言与异步生态基本概念
- 最基本的traits，类型和函数，future是由标准库提供的
- `async/await` 语法 被编译器支持
- 许多实用工具类型、宏和函数由 [`furure`](https://docs.rs/futures/) 。它们可以用于任何异步 Rust 应用。
- 异步代码的执行、IO 和任务生成均由“异步”提供 运行时“，如 Tokio 和 async-std。大多数异步应用，还有一些 异步箱子，取决于具体的运行时间。参见 “ [异步生态系统”](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html) 部分，了解更多细节。

### 编译和调试
- [Rust-异步编程-编译和调试](Rust-异步编程-编译和调试.md)
### 兼容性考虑
- [Rust-异步编程-兼容性考虑](Rust-异步编程-兼容性考虑.md)
### 性能考虑

异步 Rust 的性能取决于你所使用的异步运行时的实现。尽管驱动异步 Rust 应用的运行时相对较新，但它们在大多数实际工作负载中表现异常出色。

不过，大多数异步生态系统假设多_线程_运行时。这使得单线程异步应用的理论性能优势，尤其是更便宜的同步，变得困难。

另一个被忽视的用例是[操作系统-任务分类-延迟敏感任务](操作系统-任务分类-延迟敏感任务.md)_ ，这对驱动程序、图形界面应用等很重要。这类任务依赖运行时和/或作系统支持，以便被适当调度。未来这些用例会有更好的库支持。

### 单元测试
[Rust-异步编程-单元测试-基本概念](Rust-异步编程-单元测试-基本概念.md)
###  [Futures-rs 和生态系统](https://rust-lang.github.io/async-book/part-guide/async-await.html#futures-rs-and-the-ecosystem)

TODO 上下文和历史，futures-rs 的用途 - 曾经被广泛使用，现在可能不再需要，与 Tokio 和其他运行时重叠（有时存在细微的语义差异），为什么你可能需要它（直接使用 futures，特别是编写自己的 futures、流、一些工具）
其他生态系统相关内容——Yosh 的 crate、替代运行时、实验性内容，以及其他？
### 阻塞和取消
- 这些概念并非局限于任何特定特性或功能，而是系统普遍存在的属性，理解它们才能编写正确的代码。
#### 关于阻塞IO
- 务必确保**异步任务只能使用非阻塞 I/O，切勿使用阻塞 I/O（Rust 标准库中只提供了阻塞 I/O）**。

- 为什么？
	- 在异步中使用阻塞IO，当一个线程被阻塞意味**很多任务都必须等待**
		- 着当一个线程被阻塞时，操作系统知道不要调度它，以便其他线程可以继续执行。这在多线程程序中是可以接受的，因为这样可以让其他线程在阻塞线程等待期间继续执行。然而，在异步程序中，可能还有其他任务需要调度到同一个操作系统线程上，但操作系统并不知道这些任务的存在，因此**整个线程都会被阻塞**。这意味着，不是单个任务等待其 I/O 操作完成（这没问题），而是**很多任务都必须等待**（这就不合理了）。

#### 关于阻塞计算

**通过执行计算来阻塞线程（这与阻塞 I/O 并不完全相同，因为操作系统并未参与其中，但效果类似）。**
- 如果你有一个长时间运行的计算（无论是否阻塞 I/O），并且没有将控制权交给运行时，那么该任务将永远无法让运行时调度器有机会调度其他任务。记住，异步编程使用的是协作式多任务处理。在这种情况下，某个任务没有与其他任务协作，因此其他任务将没有机会完成工作。我们稍后会讨论如何缓解这个问题。
#### 取消

[Rust-异步编程-异步任务-取消](Rust-异步编程-异步任务-取消.md)

### 异步代码块
[Rust-异步编程-async-代码块](../Async和Await/Rust-异步编程-async-代码块.md)

###  [Async closures  异步闭包](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#async-closures)

- https://github.com/rust-lang/rust/pull/132706，https://blog.rust-lang.org/inside-rust/2024/08/09/async-closures-call-for-testing.html
- 闭包中的异步代码块与异步闭包
### [Lifetimes and borrowing  生命与借贷](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#lifetimes-and-borrowing)
- 上文提到了静态寿命。
- futures的生命周期界限（ `Future + '_` 等）
- Borrowing across await points
- 我不知道，但我确信异步函数肯定还有更多生命周期方面的问题……

### [`Send + 'static` 边界](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#send--static-bounds-on-futures)

- 它们存在的意义是什么？多线程运行时
- 生成本地文件以避免它们
- 是什么让异步函数 `Send + 'static` ，以及如何修复与之相关的错误
### [Async traits  异步特性](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#async-traits)
- syntax
	- `Send + 'static` 问题及其解决方法
		- trait_variant  性状变异
		- explicit future  明确的未来
		- return type notation 返回类型表示法https://blog.rust-lang.org/inside-rust/2024/09/26/rtn-call-for-testing.html
- 覆盖
	- 方法的 future 表示法与 async 表示法的区别
-  object safety  物体安全
- 捕获规则（https://blog.rust-lang.org/2024/09/05/impl-trait-capture-rules.html）
- history and async-trait crate

### [Recursion  递归](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#recursion)
- 允许（相对较新），但需要一些明确的装箱操作。
- 前向参考期货，固定
- https://rust-lang.github.io/async-book/07_workarounds/04_recursion.html
-  https://blog.rust-lang.org/2024/03/21/Rust-1.77.0.html#support-for-recursion-in-async-fn
- 异步递归宏（ (https://docs.rs/async-recursion/latest/async_recursion/)

### 应用经验
- [Rust-异步编程-join!](../语法/Rust-异步编程-join!.md)
- [Rust-异步编程-await-await!](../Async和Await/Rust-异步编程-await-await!.md)

## 4. 与其他卡片的关联  
- 前置卡片：
	- [操作系统-异步编程-基本概念](操作系统-异步编程-基本概念.md)
- 后续卡片：
	- [Rust-异步编程-运行时-基本概念](Rust-异步编程-运行时-基本概念.md)
- 相似主题：
	- [阻塞是什么](../../../../../../../../Zettelkasten/permanent/阻塞是什么.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
