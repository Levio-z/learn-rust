---
tags:
  - permanent
---
## 1. 核心观点  

- **Future** 是单值异步计算，表达式 `.await` 后结束，不再访问。
    
- **Stream** 是连续异步序列，需要变量能反复借用可变引用才能完成多次轮询。

`stream` Trait类似于`future` ，但在完成前可以产生多个值，类似于标准库中的`Iterator`者特征：

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

```rust
trait Stream {
    /// The type of the value yielded by the stream.
    type Item;

    /// Attempt to resolve the next item in the stream.
    /// Returns `Poll::Pending` if not ready, `Poll::Ready(Some(x))` if a value
    /// is ready, and `Poll::Ready(None)` if the stream has completed.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Self::Item>>;
}
```

### poll_next
尝试解析流（Stream）中的下一个条目。

如果尚未就绪，则返回 `Poll::Pending`；
如果值已就绪，则返回 `Poll::Ready(Some(x))`；
如果流已经处理完毕（结束），则返回 `Poll::Ready(None)`。

一个常见的流例子是future箱中`通道类型的`接收器。每次发送端发送值时，它会输出 `Some（val），` 发送方被丢弃且所有待处理消息都已接收后，则会显示 `None`：
```rust
async fn send_recv() {
    const BUFFER_SIZE: usize = 10;
    let (mut tx, mut rx) = mpsc::channel::<i32>(BUFFER_SIZE);

    tx.send(1).await.unwrap();
    tx.send(2).await.unwrap();
    drop(tx);

    // `StreamExt::next` is similar to `Iterator::next`, but returns a
    // type that implements `Future<Output = Option<T>>`.
    assert_eq!(Some(1), rx.next().await);
    assert_eq!(Some(2), rx.next().await);
    assert_eq!(None, rx.next().await);
}
```




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-异步编程-StreamExt-基本概念](Rust-异步编程-StreamExt-基本概念.md)
	- [Rust-stream-同时处理流中的多个项目](Rust-stream-同时处理流中的多个项目.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 - [x] 处理streamext 需要在vscode中编写流
 - [x] 处理同时处理流中的多个项目，使用vscode写demo
  
