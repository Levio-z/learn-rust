---
tags:
  - note
---
## 1. 核心观点  

- 推进future：通过调用`pull`函数可以推进未来，这将推动未来尽可能接近完成。如果 future 完成，则返回 `Poll：：Ready（结果）。` 如果未来尚未完成，它返回 `Poll：:P ending`，表示没有可用数据时，我们必须注册`wake`并安排在`Future`准备继续推进时调用 `wake（）` 函数。当调用 `wake（）` 时，驱动`Future`执行者会再次调用`poll` ，以便`Future`能够取得更多进展。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

- `self: Pin<&mut Self>`
	- 其中，前者的行为类似于普通的 `&mut self` 引用，不同之处在于 `Self` 值是被 [_pinned_](https://doc.rust-lang.org/nightly/core/pin/index.html) 在其内存位置。创造出不可改变的future，解决自引用问题。

- 参数 `cx: &mut Context` 
	- 的作用是**传递一个[`Waker`](https://doc.rust-lang.org/nightly/core/task/struct.Waker.html) 实例给异步任务**，让主任务知道在 `Future` 就绪时（完成）会收到通知，因此它不需要反复调用 `poll` 方法。简单理解就是context包含一个`fn（）` 函数指针，以及存储关于哪个`Future` 调用`wake`的数据。

>为什么需要wake：没有 `wake（），` 执行人将无法知道某个未来何时能取得进展，必须不断轮询每个未来。通过 `wake（），` 执行者准确知道哪些未来已准备好进行`poll` 。


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-组合设计的思想-基本概念](../组合子/Rust-future-组合设计的思想-基本概念.md)
	- [Rust-future-状态机](../Rust-future-状态机.md)
	- [Rust-Pinning-基本概念](../../Pin/Rust-Pinning-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
