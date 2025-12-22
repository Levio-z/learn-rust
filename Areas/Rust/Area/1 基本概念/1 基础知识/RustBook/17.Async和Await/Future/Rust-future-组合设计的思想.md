---
tags:
  - note
---

## 1. 核心观点  

它们可以组合成新的、“更大”的future


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
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

### Future 组合器

允许将多个 future 链式组合在一起，类似于 [`Iterator`](https://doc.rust-lang.org/stable/core/iter/trait.Iterator.html) trait的方法。 这些组合器不会等待 future 完成，而是返回一个新的 future，该 future 会在 `poll` 时应用映射操作
```rust
struct StringLen<F> {
    inner_future: F,
}

impl<F> Future for StringLen<F> where F: Future<Output = String> {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        match self.inner_future.poll(cx) {
            Poll::Ready(s) => Poll::Ready(s.len()),
            Poll::Pending => Poll::Pending,
        }
    }
}

fn string_len(string: impl Future<Output = String>)
    -> impl Future<Output = usize>
{
    StringLen {
        inner_future: string,
    }
}

// Usage
fn file_len() -> impl Future<Output = usize> {
    let file_content_future = async_read_file("foo.txt");
    string_len(file_content_future)
}
```



这段代码不完全能工作，因为它没有处理 [_pinning_](https://doc.rust-lang.org/stable/core/pin/index.html) 问题，但作为示例已经足够

- `string_len` 的目标是 **把一个返回 `String` 的 Future 转换成一个返回字符串长度的 Future**。它通过“包装（wrap）”技术把已有 future 再封装到一个结构体里（如 `StringLen<F>`），并让这个结构体再次实现 `Future` trait，从而形成一种 **Future 组合器（combinator）**。

这种模式与 `map`、`then`、`and_then` 在异步生态中的作用类似：

> **把一个 future 的输出通过某种逻辑转换成另一个 future 的输出。**



由于该函数再次返回一个 `Future` ，调用者无法直接使用返回值，而是需要再次使用组合器函数。这样一来，整个调用链就变成了异步的，我们可以在某些节点（例如 main 函数中）高效地同时等待多个 future。

>**只要链条中有一个异步点，整个链条都变成异步。**

### 组合Future的方法
#### await
[Rust-异步编程-await-基本概念](../Async和Await/Rust-异步编程-await-基本概念.md)

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
- [ ] 验证这个观点的边界条件  
