---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

Rust 目前仅提供编写异步代码所需的基本功能。值得注意的是，执行器、任务、反应器、组合器以及底层 I/O 的 future 和 traits 尚未包含在标准库中。与此同时，社区提供的异步生态系统弥补了这些不足。


### Ⅱ. 实现层

### Ⅲ. 原理层



## 2. 背景/出处  
- 来源：
	- [https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### [Async Runtimes  异步运行时](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#async-runtimes)

异步运行时是用于执行异步应用程序的库。运行时通常将一个 _Reactor_ 与一个或多个 _Executor_ 捆绑在一起。
- Reactor 提供外部事件的订阅机制，例如异步 I/O、进程间通信和定时器。在异步运行时中，Reactor通常是代表底层 I/O 操作的 Future。
- Executor 负责任务的调度和执行。它们跟踪正在运行和已挂起的任务，轮询 Future 直至完成，并在任务可以继续执行时唤醒它们。“Executor”一词经常与“运行时”互换使用。在这里，我们使用“生态系统”一词来描述捆绑了兼容特性和功能的运行时。

### [社区提供的异步 Crate](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#community-provided-async-crates)
#### [The Futures Crate  期货箱](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#the-futures-crate)
- [`futures` crate](https://docs.rs/futures/) 包含用于编写异步代码的 trait 和函数，其中包括 `Stream` 、 `Sink` 、 `AsyncRead` 和 `AsyncWrite` trait，以及诸如组合器之类的实用工具。这些实用工具和 trait 最终可能会成为标准库的一部分。
- `futures` 有自己的执行器，但没有自己的反应器，因此不支持异步 I/O 或定时器 futures 的执行。正因如此，它不被视为一个完整的运行时环境。常见的做法是使用 `futures` 中的工具函数，并搭配来自其他 crate 的执行器。

#### [Popular Async Runtimes  常用的异步运行时](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#popular-async-runtimes)

- [Tokio](https://docs.rs/tokio/) ：一个流行的异步生态系统，包含 HTTP、gRPC 和跟踪框架。
- [async-std](https://docs.rs/async-std/) ：一个为标准库组件提供异步对应项的 crate。
- [smol](https://docs.rs/smol/) ：一个小型、简化的异步运行时。提供可用于包装 `UnixStream` 或 `TcpListener` 等结构体的 `Async` trait。
- [fuchsia-async](https://fuchsia.googlesource.com/fuchsia/+/master/src/lib/fuchsia-async/) ：Fuchsia 操作系统中使用的执行器。

### [确定生态系统兼容性](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#determining-ecosystem-compatibility)

- 并非所有异步应用程序、框架和库都彼此兼容，也并非所有操作系统或平台都兼容。**大多数异步代码可以与任何生态系统一起使用，但有些框架和库需要使用特定的生态系统。生态系统限制并非总是有文档记录，但有一些经验法则可以用来判断某个库、特性或函数是否依赖于特定的生态系统。**
- 任何**与异步 I/O、定时器、进程间通信或任务交互的异步代码通常都依赖于特定的异步执行器或反应器**。所有其他异步代码，例如异步表达式、组合器、同步类型和流，通常都是生态系统无关的，前提是任何嵌套的 Future 也与生态系统无关。在开始一个项目之前，建议研究相关的异步框架和库，以确保它们与你选择的运行时以及彼此之间的兼容性。

- 值得注意的是， `Tokio` 使用了 `mio` reactor，并定义了自己的异步 I/O trait版本，包括 `AsyncRead` 和 `AsyncWrite` 。它本身与依赖于 [`async-executor` crate 的](https://docs.rs/async-executor) `async-std` 和 `smol` 不兼容，而 async-std 和 smol 又依赖于  `futures` 中定义的`AsyncRead` 和 `AsyncWrite`特征。

- 有时可以通过兼容层来解决运行时冲突，兼容层允许您在一个运行时环境中调用为另一个运行时环境编写的代码。例如， [`async_compat` crate](https://docs.rs/async_compat) 提供了一个兼容层，用于在不同的运行时环境之间进行交互。 `Tokio` 和其他运行时环境。

- 除非需要生成任务或**定义自己的异步 I/O 或定时器 Future**，否则提供异步 API 的库不应依赖特定的执行器或反应器。理想情况下，**只有二进制文件才应负责调度和运行任务**。

### [单线程执行器与多线程执行器](https://rust-lang.github.io/async-book/08_ecosystem/00_chapter.html#single-threaded-vs-multi-threaded-executors)

异步执行器可以是单线程的，也可以是多线程的。例如， `async-executor` crate 同时包含单线程的 `LocalExecutor` 和多线程的 `Executor` 。

多线程执行器可以同时处理多个任务。**对于任务量大的工作负载，它可以显著加快执行速度，但任务间的数据同步通常开销更大**。在选择单线程运行时还是多线程运行时时，建议对应用程序的性能进行评估。

任务既可以在创建它的线程上运行，也可以在单独的线程上运行。异步运行时通常提供将任务派生到单独线程的功能。即使任务在单独的线程上执行，它们也应该是非阻塞的。为了将任务调度到多线程执行器上，它们还必须是 `Send` 。**一些运行时提供了用于派生非 `Send` 任务的函数，这确保每个任务都在创建它的线程上执行**。它们也可能提供用于将阻塞任务派生到专用线程的函数，这对于运行来自其他库的阻塞同步代码非常有用。


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
