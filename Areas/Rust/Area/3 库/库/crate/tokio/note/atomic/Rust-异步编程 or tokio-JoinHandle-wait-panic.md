---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

- 如果任务发生 panic 或被中止（一种[取消](https://rust-lang.github.io/async-book/part-reference/cancellation.html)方式），则结果为包含 [`JoinError`](https://docs.rs/tokio/latest/tokio/task/struct.JoinError.html) `Err` 对象（详见文档）。

### 传播panic
- 如果你的项目中不调用 `abort`，那么 `JoinError` **只可能来源于 panic**。
- 在这种情况下，调用 `unwrap()` 相当于把任务内部 panic **传播到生成任务**，逻辑上与同步线程类似：
- 任务 panic → `unwrap()` → 当前任务 panic
>**这样做可以让 panic 不被吞掉，保持错误透明性，符合 Rust 的安全和可预测原则。**
### Ⅱ. 应用层



### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### `JoinHandle.await` 与 panic/取消行为解释

在 Tokio 中，使用 `tokio::spawn` 生成的任务会返回一个 `JoinHandle`，调用 `.await` 可以获取任务的执行结果。理解它的行为需要注意两种特殊情况：任务 panic 与任务取消。

---

#### 1. 任务发生 panic
- 如果任务在执行过程中 panic，任务不会直接中止整个程序（除非是主线程 panic），而是被 Tokio 捕获。
- 此时，`JoinHandle.await` 会返回一个 `Result<T, JoinError>`：
    - `Ok(T)`：任务正常完成，返回结果 `T`
    - `Err(JoinError)`：任务 panic（或被取消）
- 这意味着 `.await` **将任务的 panic “传播”到调用者**，你可以选择处理它或使用 `unwrap()` 让 panic 继续传播：

```rust
use tokio::task::spawn;

#[tokio::main]
async fn main() {
    let handle = spawn(async {
        panic!("something went wrong");
    });

    // propagate panic to the current task
    handle.await.unwrap(); // 这里会 panic
}
```

- 这种传播机制类似于线程中的 `thread::JoinHandle::join()`。
    

---

#### 2. 任务被取消（abort）
- Tokio 允许任务被显式取消（abort），例如 `handle.abort()`。
- 如果任务被取消，再 `.await` 会返回 `Err(JoinError)`，表示任务未完成。
- 文档参考：[Async cancellation](https://rust-lang.github.io/async-book/part-reference/cancellation.html)

```rust
use tokio::task::spawn;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let handle = spawn(async {
        sleep(Duration::from_secs(10)).await;
        42
    });

    handle.abort(); // 任务被取消

    match handle.await {
        Ok(val) => println!("task finished: {}", val),
        Err(err) => println!("task failed or cancelled: {:?}", err),
    }
}
```
## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-tokio-基本使用](Rust-tokio-基本使用.md)
	- [Rust-Async和Await-基本概念](../../../../../../1%20基本概念/2%20进阶/2.3%20并发和异步/Async和Await/Rust-Async和Await-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
