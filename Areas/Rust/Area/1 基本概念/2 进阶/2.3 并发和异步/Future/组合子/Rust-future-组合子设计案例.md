---
tags:
  - note
---

## 1. 核心观点  

允许将多个 future 链式组合在一起，类似于 [`Iterator`](https://doc.rust-lang.org/stable/core/iter/trait.Iterator.html) trait的方法。 这些组合器不会等待 future 完成，而是返回一个新的 future，该 future 会在 `poll` 时应用映射操作


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

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

**如果涉及借用和不同的生命周期，情况会变得更复杂。因此，Rust 投入了大量工作，增加了async/wait支持，目标是让异步代码的组合编写过程极为简单。**


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
  
