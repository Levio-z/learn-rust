---
tags:
  - note
---

## 1. 核心观点  

在同步代码中，函数 `f()` 返回 `i32`，你可以直接拿来加减。但在异步代码中，`f()` 返回的是 `Future<Output = i32>`。

- 如果你想在函数 `g()` 里使用这个 `i32`，`g()` 自己也必须变成异步的（或者使用组合器）。
    
- 这导致了**异步调用链**：为了处理一个底层的小 Future，你外层的所有函数都会被“卷入”，最终变成一个巨大的复合 Future。

**否则就需要实现轮询和线程阻塞，外面不是future就需要阻塞或线程的概念来执行。**同步里面不兼容异步，保证不降级为同步代码。

由于该函数再次返回一个 `Future` ，调用者无法直接使用返回值，而是需要再次使用组合器函数。这样一来，整个调用链就变成了异步的，我们可以在某些节点（例如 main 函数中）高效地同时等待多个 future。

>**只要链条中有一个异步点，整个链条都变成异步。**


### 等待future就绪
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
等待 Future 变为就绪状态
```rust
let future = async_read_file("foo.txt");
let file_content = loop {
    match future.poll(…) {
        Poll::Ready(value) => break value,
        Poll::Pending => {}, // 什么都不做
    }
}
```

分析：方案可行，但效率非常低下，因为它会让 CPU 持续忙碌直到值变得可用。

更高效的方法可能是 _阻塞_ 当前线程，直到 future 值变得可用
- 这只有在拥有线程的情况下才可能实现，因此该解决方案不适用于我们的内核，至少目前还不适用。
- 即使在支持阻塞的系统上，这种方式通常也不被推荐，因为它会将异步任务再次转变为同步任务，从而抑制了并行任务潜在的性能优势。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-异步编程-await-基本概念](../../Async和Await/Rust-异步编程-await-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  
