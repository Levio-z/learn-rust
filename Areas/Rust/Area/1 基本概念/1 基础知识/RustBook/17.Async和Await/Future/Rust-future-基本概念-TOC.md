---
tags:
  - note
---

## 1. 核心观点  

`Future` 代表延迟计算（一个可能还没有准备好的值），但会在未来的某个时间点完成。Rust 中异步并发的基本单元是 _future。future_ 就是一个普通的 Rust 对象（通常是结构体或枚举），它实现了 [`Future`](https://doc.rust-lang.org/std/future/trait.Future.html) trait

> 您可以认为它是一种延迟的结果，它在未来的某个时间点会完成（比如网络请求的响应、磁盘读写操作等）。



### 优点
future 组合器的**最大优势在于它们能保持操作的异步性**，从而实现高性能。
- **方式**：与异步 I/O 接口结合，实现极高的性能。
- **编译器深度优化**：future 组合器以普通结构体配合 trait 的方式实现，使得编译器能够对其进行深度优化。

### 缺点
- 复杂性：
	- 虽然 future 组合器能够编写出非常高效的代码，但在某些情况下，由于类型系统和基于闭包的接口，它们可能变得难以使用。易于产生**生命周期问题和类型问题**。
	- 案例：[[# 复杂性：代码块中返回了不同的 future 类型，必须使用包装器类型将它们统一为单一类型]]
### 特性
### 惰性对象
关于 Rust 中的 future，一个重要的直觉是它们是**惰性对象。要执行任何操作，它们必须由外部力量（通常是异步运行时）驱动**。
- 它不会立即执行函数中的代码。此外，Future 对象只有在被wait后才会执行任何操作。这与其他一些语言不同，在这些语言中，异步函数返回的 Future 对象会立即开始执行。
### 传染性
**只要链条中有一个异步点，整个链条都变成异步。** 详情见[Rust-future-组合设计的思想](Rust-future-组合设计的思想.md)
#### 组合设计
- [Rust-future-组合设计的思想](Rust-future-组合设计的思想.md)

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 概念
每个实现 Future 的类型 包含有关已取得的进展以及“就绪”内容的信息 方法。 
1. **进展信息**（当前状态）
	就是这个 Future 的当前状态（状态机在哪一步）：
	- `Start`：还没开始执行
	- `WaitingAdd`：正在等另一个 Future 完成
	- `Done`：已完成，等待用户取值
2. **“就绪”内容信息**（结果缓存）
	- `result: Option<usize>` 这个字段就用来缓存最终的输出结果；
	- 一旦 `poll()` 返回 `Poll::Ready(result)`，就说明 Future 已就绪。
### 源码
```rust
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>;
}
```

- [关联类型](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#associated-types) `Output` 用于指定异步值的类型。例如, 上图中的 `async_read_file` 函数将返回一个 `Future` 实例，其 `Output` 被设置为 `File`
- [`poll`](https://doc.rust-lang.org/nightly/core/future/trait.Future.html#tymethod.poll) 方法可用于检查值是否已就绪。它返回一个 [`Poll`](https://doc.rust-lang.org/nightly/core/future/trait.Future.html#tymethod.poll) 枚举，其定义如下：

```rust
pub enum Poll<T> {
    Ready(T),
    Pending,
}
```
当值已可用时（例如文件已从磁盘完全读取），它会被包装后返回 `Ready` 变体。否则返回 `Pending` 变体，向调用者表明该值尚不可用。

`poll` 方法接收两个参数： `self: Pin<&mut Self>` and `cx: &mut Context` 。
- 其中，前者的行为类似于普通的 `&mut self` 引用，不同之处在于 `Self` 值是被 [_pinned_](https://doc.rust-lang.org/nightly/core/pin/index.html) 在其内存位置。
- 如果不了解 async/await 的工作原理，那么理解 `Pin` 的原理和必要性会变得很困难。因此我们会在后文中详细解释。

参数 `cx: &mut Context` 的作用是传递一个[`Waker`](https://doc.rust-lang.org/nightly/core/task/struct.Waker.html) 实例给异步任务，例如文件系统加载。**这个 `Waker` 允许异步任务发出信号来表明它已全部或者部分完成，例如文件已从磁盘加载完成。由于主任务知道在 `Future` 就绪时它会收到通知，因此它不需要反复调用 `poll` 方法**。我们将在本文后面实现自己的 waker 类型时更详细地解释这个过程。


### 一个简单的例子来理解
- 来源：https://os.phil-opp.com/zh-CN/async-await/
![](asserts/Pasted%20image%2020251211153537.png)
该序列图展示了一个 `main` 函数，它从文件系统中读取文件，然后调用 `foo` 函数。这个过程会重复两次：一次使用同步的 `read_file` 调用，另一次使用异步的 `async_read_file` 调用。

使用同步调用时， `main` 函数需要等待文件从文件系统中加载完成后才能调用 `foo` 函数。

通过异步的 `async_read_file` 调用，**文件系统会直接返回一个 future 并在后台异步加载文件**。这使得 `main` 函数能够更早地调用 `foo` ，然后 `foo` 会与文件加载并行运行。在这个例子中，文件加载甚至在 `foo` 返回前就完成了，因此 `main` 在 `foo` 返回后无需等待就能直接处理文件。
### 复杂性：代码块中返回了不同的 future 类型，必须使用包装器类型将它们统一为单一类型
```rust
fn example(min_len: usize) -> impl Future<Output = String> {
    async_read_file("foo.txt").then(move |content| {
        if content.len() < min_len {
            Either::Left(async_read_file("bar.txt").map(|s| content + &s))
        } else {
            Either::Right(future::ready(content))
        }
    })
}
```

这里我们读取 `foo.txt` 文件，然后使用 `then` 组合器根据文件内容链接第二个future。如果内容长度小于给定的 `min_len`，我们会读取另一个文件 `bar.txt` 并将其追加到 `content` ，否则仅返回 `foo.txt` 的内容。

我们需要对传递给 `then` 的闭包使用 [move 关键字](https://doc.rust-lang.org/std/keyword.move.html)，否则 `min_len` 中会出现生命周期错误。使用 [`Either`](https://docs.rs/futures/0.3.4/futures/future/enum.Either.html) 包装器的原因是 `if` 和 `else` 代码块必须始终保持相同的类型。**由于我们在代码块中返回了不同的 future 类型，必须使用包装器类型将它们统一为单一类型**。[`ready`](https://docs.rs/futures/0.3.4/futures/future/fn.ready.html) 函数将一个值包装成立刻可用的 future。这里需要该函数是因为 `Either` 包装器要求被包装的值必须实现 Future。

可以想象，对于大型项目来说，这很快就会导致代码变得复杂。特别是涉及借用和不同的生命周期时，情况会变得更加复杂。正因如此，大量工作被投入到为 Rust 添加 async/await 支持中，来让异步代码编写起来更简单。

### Pinning 与 Futures
正如我们在这篇文章中已经看到的，[`Future::poll`](https://doc.rust-lang.org/nightly/core/future/trait.Future.html#tymethod.poll) 方法通过 `Pin<&mut Self>` 参数的形式使用固定：

```rust
fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>
```

该方法采用 `self: Pin<&mut Self>` 而非普通的 `&mut self` 的原因是，**通过 async/await 创建的 future 实例通常是自引用的，如我们[之前](https://os.phil-opp.com/async-await/#self-referential-structs)所见的那样**。将 `Self` 包装进 `Pin` 并让编译器为 async/await 生成的自引用 future 不实现 `Unpin` ，可以确保在 `poll` 调用之间这些 future 在内存中不会被移动。这确保了所有内部引用仍然有效。

值得注意的是，在首次调用 `poll` 前移动 future 是安全的。这是由于 future 有惰性，在首次被轮询前不会执行任何操作。刚生成的状态机处于 `start` 状态，因此仅包含函数参数而不包含内部引用。为了调用 `poll` ，调用者必须先将 future 包装到 `Pin` 中，这确保了 future 在内存中不再能被移动。由于栈固定（stack pinning）更难实现，我建议在这种情况下始终结合使用 [`Box::pin`](https://doc.rust-lang.org/nightly/alloc/boxed/struct.Box.html#method.pin) 和 [`Pin::as_mut`](https://doc.rust-lang.org/nightly/core/pin/struct.Pin.html#method.as_mut)。

如果你有兴趣了解如何安全地使用栈固定（pinning）技术自行实现一个 future 组合器函数，可以参考 `futures` crate 中相对简短的 [map 组合器 方法的源代码](https://docs.rs/futures-util/0.3.4/src/futures_util/future/future/map.rs.html) 以及 pin 文档中关于 [projections and structural pinning](https://doc.rust-lang.org/stable/std/pin/index.html#projections-and-structural-pinning) 的章节。

### 本质
协作式方法的最大优势在于任务能够自行保存状态，从而实现更高效的上下文切换，并允许任务间共享同一个调用栈。

虽然可能不太明显，但 futures 和 async/await 实际上是一种协作式多任务模式的实现：

- 每个添加到执行器的 future 本质上都是协作式任务。
- 相对于使用显式的 yield 操作符，future 通过 `Poll::Pending`（或在最后 `Poll::Ready`）放弃 CPU 核心的控制权。
    - 并没有谁要强制 future 放弃 CPU。如果它们想，它们可以永不从 `poll` 中返回。例如通过无限循环。
    - 由于每个 future 都有能力阻断执行器中其他 future 的执行，我们得首先相信它们是无恶意的。
- Future 内部存储了所有在下一次 `poll` 调用时继续执行所需的状态。使用 async/await 时，编译器会自动检测所有需要的变量并将它们存储在生成的状态机内部。
    - 仅保存继续执行所需的最小状态。
    - 由于 `poll` 方法**在返回时会释放调用栈，这同一个栈可以用于轮询其他 future**。

我们看到 future 和 async/await 完美契合协作式多任务模式；它们只不过使用了一些不同的术语。因此在下文中，术语 “任务 task” 和 “future” 可以互换使用。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-组合设计的思想](Rust-future-组合设计的思想.md)
	- [Rust-future-状态机](Rust-future-状态机.md)
	- [Rust-Pinning-基本概念](../Pin/Rust-Pinning-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
