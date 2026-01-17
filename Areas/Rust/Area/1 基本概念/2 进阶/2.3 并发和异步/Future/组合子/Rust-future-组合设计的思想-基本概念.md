---
tags:
  - note
---

## 1. 核心观点  

它们可以组合成新的、“更大”的future，下面这些示例展示了`future tait`如何用来表达异步控制流，而无需多个分配对象和深度嵌套回调。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
该 `Future`s 模型允许组合多个异步作而无需中间分配。同时运行多个未来或将未来串联，可以通过无分配状态机实现，如下：

```rust
/// 一个简单的 Future（SimpleFuture），它能并发地运行另外两个 Future 直到它们完成。
///
/// 这里的“并发”是通过以下事实实现的：对每个 Future 的 `poll`（轮询）调用可以是交错进行的，
/// 从而允许每个 Future 按照各自的节奏推进。
pub struct Join<FutureA, FutureB> {
    // 每个字段可能包含一个需要运行至完成的 Future。
    // 如果某个 Future 已经完成，该字段将被设置为 `None`。
    // 这可以防止我们在 Future 完成后再次对其进行 poll，
    // 否则将违反 `Future` 特性（trait）的约定。
    a: Option<FutureA>,
    b: Option<FutureB>,
}

impl<FutureA, FutureB> SimpleFuture for Join<FutureA, FutureB>
where
    FutureA: SimpleFuture<Output = ()>,
    FutureB: SimpleFuture<Output = ()>,
{
    type Output = ();
    fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
        // 尝试推进 Future `a` 的完成进度。
        if let Some(a) = &mut self.a {
            if let Poll::Ready(()) = a.poll(wake) {
                // 如果完成了，就把 Future 从 Option 中取出并丢弃
                self.a.take();
            }
        }

        // 尝试推进 Future `b` 的完成进度。
        if let Some(b) = &mut self.b {
            if let Poll::Ready(()) = b.poll(wake) {
                // 同理，如果完成了，就将其设为 None
                self.b.take();
            }
        }

        if self.a.is_none() && self.b.is_none() {
            // 两个 Future 都已经完成了 —— 我们可以成功返回 Ready
            Poll::Ready(())
        } else {
            // 其中一个或两个 Future 返回了 `Poll::Pending`，仍有工作要做。
            // 当它们可以继续推进时，会调用传入的 `wake()` 函数。
            Poll::Pending
        }
    }
}
```

这表明多个未来可以同时运行，无需单独分配，从而实现更高效的异步程序。类似地，多个线性future也可以连续运行，如下：
```rust
/// 一个简单的 Future（SimpleFuture），它会按顺序运行两个 Future 直到完成。
///
/// 注意：为了简化示例，此处的 `AndThenFut` 假设第一和第二个 Future 
/// 在创建时都是已知的（可用的）。而真正的 `AndThen` 组合器允许
/// 根据第一个 Future 的输出来创建第二个 Future，
/// 就像 `get_breakfast.and_then(|food| eat(food))` 这样。
pub struct AndThenFut<FutureA, FutureB> {
    first: Option<FutureA>,
    second: FutureB,
}

impl<FutureA, FutureB> SimpleFuture for AndThenFut<FutureA, FutureB>
where
    FutureA: SimpleFuture<Output = ()>,
    FutureB: SimpleFuture<Output = ()>,
{
    type Output = ();
    fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
        // 如果第一个 Future 还没完成（依然在 Option 中）
        if let Some(first) = &mut self.first {
            match first.poll(wake) {
                // 我们已经完成了第一个 Future —— 将其移除并开始第二个！
                Poll::Ready(()) => {
                    self.first.take();
                }
                // 我们暂时还无法完成第一个 Future。
                // 注意，我们通过 `return` 语句中断了 `poll` 函数的流程（直接返回 Pending）。
                Poll::Pending => return Poll::Pending,
            };
        }
        
        // 既然第一个 Future 已经做完了，现在尝试完成第二个任务。
        self.second.poll(wake)
    }
}
```
### 解决了什么问题，如果没有组合子
- [Rust-future-组合设计传染性](Rust-future-组合设计传染性.md)
- [Rust-future-组合子设计思想](Rust-future-组合子设计思想.md)
	- [Rust-future-组合子设计案例](Rust-future-组合子设计案例.md)
	- async/wait就是rust投入大量工作让异步的组合子使用更简单的成果。
		- **如果涉及借用和不同的生命周期，情况会变得更复杂。因此，Rust 投入了大量工作，增加了async/wait支持，目标是让异步代码的组合编写过程极为简单。**
- [Rust-future-组合子高效同时等待的思维模型](Rust-future-组合子高效同时等待的思维模型.md)
	- [Rust-future-解耦做什么和什么时候做](Rust-future-解耦做什么和什么时候做.md)





## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-异步编程-await-基本概念](../../Async和Await/Rust-异步编程-await-基本概念.md)
- 后续卡片：
	- [Rust-future-组合子设计思想](Rust-future-组合子设计思想.md)
	- [Rust-future-组合子设计案例](Rust-future-组合子设计案例.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  
