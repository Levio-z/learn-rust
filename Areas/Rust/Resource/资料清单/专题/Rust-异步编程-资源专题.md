---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

- `async` ：`async` 是函数（以及其他元素，例如 trait，我们稍后会讲到）上的注解，可以应用于块和函数，以指定它们可以被中断和恢复。`sync` 关键字可用于函数签名中来将一个同步函数转换为返回 future 的异步函数。
    - 【异步块】当 Rust 看到一个标有 `async` 关键字的块时，它会将其编译成一个唯一的匿名数据类型，实现 `Future` trait。
    - [Rust-异步编程-async-异步函数](../../../Area/1%20基本概念/2%20进阶/2.3%20并发和异步/Async和Await/Rust-异步编程-async-异步函数.md)
- 在编写异步 Rust 时，我们使用 `async` 和 `await` 关键字。 Rust 使用 `Future` trait 将它们编译成等效代码，就像它使用 `Iterator` trait 将 `for` 循环编译成等效代码一样。

### 调用者角度

**使用 `async` 关键字声明的函数，这意味着它可以异步执行**。也就是说，调用者可以选择不等待函数执行完毕就执行其他操作。

### 执行方式

- **代码顺序**：在异步函数内部，代码的执行方式与通常的顺序执行方式相同 [¹](https://rust-lang.github.io/async-book/part-guide/async-await.html#footnote-preempt) ，异步本身并无区别。你可以从异步函数中调用同步函数，执行过程也与往常一样。

>	在未引入数据竞争的前提下，执行结果等价于某个顺序执行。**没有读取**可能被其他线程修改的共享数据，或**读取的是受同步原语保护**的数据（锁、原子、happens-before 关系），那么即便发生过抢占，函数也**无法区分**“是否被暂停过”。这源于语言与硬件提供的**顺序一致性/内存模型抽象**：在未引入数据竞争的前提下，执行结果等价于某个顺序执行。

- **使用await等待其他异步函数**：异步函数中还可以使用 `await` 来等待其他异步函数（或 Future）执行完毕，这可能会导致控制权的释放，以便其他任务可以执行。

### Ⅱ. 实现层


### Ⅲ. 原理层

当调用异步函数时，其函数体不会像普通函数那样执行。相反，函数体及其参数会被打包到一个 Future 对象中，并返回该 Future 对象而不是实际结果。调用者可以决定如何处理这个 Future 对象（如果调用者想要“立即”获得结果，则会 `await` 该 Future 对象，参见下一节）。


## 2. 背景/出处  
- 来源：[RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 参考资料


### 基本概念
#### async
- `async` 关键字可以应用于块和函数，以指定它们可以被中断和恢复。`sync` 关键字可用于函数签名中来将一个同步函数转换为返回 future 的异步函数。
    - 【异步块】当 Rust 看到一个标有 `async` 关键字的块时，它会将其编译成一个唯一的匿名数据类型，实现 `Future` trait。
    - 【异步函数】当 Rust 看到一个标有 `async` 的函数时，它会将其编译成一个非异步函数，其主体是一个异步块。因此，async 函数的返回类型是编译器为该 async 块创建的匿名数据类型的类型。
	    - `async fn calculate(nums: &[i32]) -> i32 {}`
	    - `fn calculate<'a>(nums: &'a [i32]) -> impl Future<Output = i32> + 'a;` 
	    - 异步函数返回一个匿名类型，该类型实现 `Future` trait，其输出是函数的返回类型。所以在这里，它表示为 `impl Future<Output = i32>`。future 捕获函数参数中的任何生命周期。因此，返回的类型具有边界 `+ 'a`，而输入切片的类型为 `&'a [i32]。` 这表明 slice 的生存时间必须至少与捕获它的 future 一样长。
- 在编写异步 Rust 时，我们使用 `async` 和 `await` 关键字。 Rust 使用 `Future` trait 将它们编译成等效代码，就像它使用 `Iterator` trait 将 `for` 循环编译成等效代码一样。

### 案例
```rust
async fn foo() -> u32 {
    0
}

// 上述代码大致被编译器转换成
fn foo() -> impl Future<Output = u32> {
    future::ready(0)
}
```


#### 零成本抽象
在编写异步 Rust 时，我们大部分时间都使用 `async` 和 `await` 关键字。Rust 使用 `Future` trait 将它们编译成等价代码，就像它使用 `Iterator` trait 将 `for` 循环编译成等价代码一样。不过，由于 Rust 提供了 `Future` trait，因此您也可以在需要时为自己的数据类型实现它。

前者案例见：
[Rust-future-组合设计的思想-基本概念](../../../Area/1%20基本概念/2%20进阶/2.3%20并发和异步/Future/组合子/Rust-future-组合设计的思想-基本概念.md)


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-异步编程-async-代码块](../../../Area/1%20基本概念/2%20进阶/2.3%20并发和异步/Async和Await/Rust-异步编程-async-代码块.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
