---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层
- 在`async fn` 中， `await` 是表达式中使用的运算符，用于获取一个 future 的异步值
	- 与 `block_on` 不同，`.await` 不会阻塞当前线程，而是异步等待未来任务完成，如果未来当前无法推进，允许其他任务运行。
### Ⅱ. 应用层
- 应用注意事项
	- 它是一个后缀关键字，与 `.` 运算符一起使用。这意味着它可以方便地用于方法调用和字段访问链中。
	- 只能在异步上下文中使用，等待future就绪[为什么只能在异步上下文中使用](#为什么只能在异步上下文中使用)
	- 使用await语句不会改变顺序语句的并发性，其本身而言，语句都是按顺序执行的，但是`await` 处可能穿插着另一个异步任务
- 是组合[future](../Future/Rust-future-基本概念-TOC.md)的一种方法
### Ⅲ. 实现层
- 实现：[await等价语义](#await等价语义)，await的底层是或者使用轮询（polled），这是一种比 `await` 更底层的操作，在使用 `await` 时会在后台执行。我们稍后在详细讨论 futures 时会谈到轮询 。
### **IV**.原理层
- await如何实现并发: [操作系统-线程并发系统中的阻塞式 IO VS 异步并发系统中的非阻塞式 IO](../../../../../../../../../Zettelkasten/permanent/操作系统-线程并发系统中的阻塞式%20IO%20VS%20异步并发系统中的非阻塞式%20IO.md)中的本质
- 本质：[设计：await就是暂停点](#设计：await就是暂停点)

## 2. 背景/出处  
- 来源：
	- [RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
	- https://rust-lang.github.io/async-book/part-guide/async-await.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 为什么只能在异步上下文中使用
[设计：await就是可能将控制权交给运行时的位置](#设计：await就是可能将控制权交给运行时的位置)

`await` 只能在异步上下文中使用，目前这意味着在异步函数中使用（我们稍后会介绍更多类型的异步上下文）。
- **运行时的抽象理解**：在异步上下文中，只有运行时可以作为控制权的传递对象。目前，您可以将运行时**想象成一个全局变量，它只能在异步函数中访问**，我们稍后会解释它的实际工作原理。
### 设计：await就是暂停点
- 暂停点也就是也就是可能将控制权交给运行时的位置，从而实现高效利用资源，来执行另一个任务
- **核心设计：实现并发的方式**：它允许当前任务继续执行，或者，如果当前任务现在无法继续执行，则允许另一个任务继续执行。
- **为了实现上面**：我们需要抽象出一个调度器，处理和任务无关的当前状态保存和处理任务的调度
- **如何做到呢**：
	- 如果结果可以立即完成，或者无需等待即可计算，那么 `await` 会直接执行计算并生成结果。
	- 但是，如果结果尚未完成，则 `await` 在当前位置把控制权交还给运行时，并保存当前计算状态，以便运行时可以启动其他任务的一些工作，然后在它已准备好再次尝试推进第一个。这是一个看不见的状态机， 就好像你写了一个这样的枚举来保存每个 await 的当前状态点。

#### await等价语义
1. 对 `fut` 调用 `poll`
2. 若返回 `Ready(v)` → 继续执行
3. 若返回 `Pending`：
    - 保存当前执行状态
    - 注册 `Waker`
    - **立刻返回 `Pending` 给调用者**
这整个过程称为 **polling（轮询）**，而不是阻塞等待。

#### 零成本抽象
在编写异步 Rust 时，我们大部分时间都使用 `async` 和 `await` 关键字。Rust 使用 `Future` trait 将它们编译成等价代码，就像它使用 `Iterator` trait 将 `for` 循环编译成等价代码一样。不过，由于 Rust 提供了 `Future` trait，因此您也可以在需要时为自己的数据类型实现它。

前者案例见：
[Rust-future-组合设计的思想-基本概念](../Future/组合子/Rust-future-组合设计的思想-基本概念.md)
```rust
async fn example(min_len: usize) -> String {
    let content = async_read_file("foo.txt").await;
    if content.len() < min_len {
        content + &async_read_file("bar.txt").await
    } else {
        content
    }
}

```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-异步编程-join!](../语法/Rust-异步编程-join!.md)
	- [Rust-异步编程-await-await!](Rust-异步编程-await-await!.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
