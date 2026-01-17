---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层
| 方式                | 取消类型 | 是否协作 | 是否 spawn 后可用 | 是否可清理 |
| ----------------- | ---- | ---- | ------------ | ----- |
| Drop Future       | 隐式   | 否    | 否            | ❌     |
| abort             | 强制   | 否    | 是            | ❌     |
| CancellationToken | 协作   | 是    | 是            | ✅     |
| select            | 隐式   | 否    | 是            | ❌     |
使用 `CancellationToken` 需要被取消的 Future 的配合，而其他方法则不需要。在其他情况下，被取消的 Future 不会收到取消通知，也没有机会进行清理（除了其析构函数）。请注意，即使 Future 拥有取消令牌，仍然可以通过其他不会触发取消令牌的方法取消它。

从编写异步代码（在异步函数、代码块、future 等中）的角度来看，代码可能会在任何 `await` 处停止执行（包括宏中的隐藏 await），并且永远不会再次执行。为了确保代码正确（特别是_确保其可取消性_ ），无论代码是正常完成还是在任何 await 语句处终止，它都必须能够正常工作 [¹](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#footnote-cfThreads) 。

>**将异步编程中的取消操作与线程取消进行比较很有意思**。取消线程是可能的（例如，在 C 语言中使用 `pthread_cancel` ，Rust 中没有直接的方法），但这几乎总是一个非常糟糕的做法，因为被取消的线程可能在任何地方终止。相比之下，取消异步任务只能在 `await` 点进行。因此，在不终止整个进程的情况下取消操作系统线程的情况非常罕见，所以作为程序员，你通常不必担心这种情况的发生。然而，在 Rust 的异步编程中，取消操作绝对是_可能_发生的。我们将在后续内容中讨论如何处理这种情况 [。](https://rust-lang.github.io/async-book/part-guide/more-async-await.html#fr-cfThreads-1)
### Ⅱ. 应用层


## 2. 背景/出处  
- 来源：
	- [RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
	- https://rust-lang.github.io/async-book/part-guide/async-await.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


### 一、通过 **Drop Future（丢弃 Future）** 实现取消

####  定义
如果你**拥有一个 `Future` 的所有权**，直接将其丢弃（drop），该 Future 将**立即停止推进**，等价于被取消。
#### 原理
- `Future` 本质是一个状态机
- 只有在被 `poll()` 时才会继续执行
- 一旦被 drop：
    - 状态机被销毁
    - 后续不会再被 `poll`
    - 不会再产生任何副作用

这是 **最底层、最原始的取消机制**。

#### 示例

```rust
use tokio::time::{sleep, Duration};

async fn work() {
    sleep(Duration::from_secs(10)).await;
    println!("done");
}

#[tokio::main]
async fn main() {
    let fut = work();
    // fut 被丢弃，永远不会执行
}
```

#### 行为特征

- ❌ **不可恢复**
- ❌ 不触发清理逻辑（除非 Drop 实现）
- ✅ 零成本
- ✅ 无需运行时支持

#### 使用场景
- Future 还未被 spawn
- 组合式 Future（如 map / then）中途放弃
- 构建高阶控制流时的“自然取消”
---

###  二、通过 **JoinHandle::abort / AbortHandle** 取消任务

#### 定义
对已经 `spawn` 到运行时中的任务，通过 `JoinHandle::abort()` 显式请求取消。
#### 原理
- 运行时为任务维护一个 **取消标志**
- `abort()`：
    - 标记任务为 cancelled
    - 下次 `.await` / yield 时任务被强制终止
- 任务不会再继续执行用户代码

#### 示例

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        sleep(Duration::from_secs(10)).await;
        println!("never runs");
    });

    handle.abort();

    let res = handle.await;
    assert!(res.is_err()); // JoinError::is_cancelled()
}
```

#### 行为特征
- ❌ 不保证运行 Drop / finally-like 逻辑
- ❌ 任务中途被“硬切”
- ✅ 即使不持有 Future 本体也能取消
- ✅ 适合运行时级别管理
#### 使用场景
- 超时控制
- 后台任务失效
- Actor / supervisor 模型
---

###  三、通过 **CancellationToken（协作式取消）**

#### 定义
使用共享的取消信号（token），**由 Future 主动检查并决定何时退出**。
#### 原理
- token 本身不会中断执行
- Future 在 `.await` 点或循环中检查 token
- 取消是 **协作式（cooperative）** 的

#### 示例

```rust
use tokio_util::sync::CancellationToken;
use tokio::time::{sleep, Duration};

async fn worker(token: CancellationToken) {
    loop {
        tokio::select! {
            _ = token.cancelled() => {
                println!("graceful shutdown");
                break;
            }
            _ = sleep(Duration::from_secs(1)) => {
                println!("working...");
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let token = CancellationToken::new();
    let child = token.child_token();

    let handle = tokio::spawn(worker(child));

    sleep(Duration::from_secs(3)).await;
    token.cancel();

    handle.await.unwrap();
}
```

#### 行为特征

- ✅ 支持资源清理
- ✅ 可控退出点
- ❌ 需要 Future 主动配合
- ❌ 有少量状态开销
    

#### 使用场景
- 服务优雅关闭
- 长循环任务
    
- 流处理 / daemon
    

---

### 四、通过 **select / select! 隐式取消**

#### 定义
在 `select` 中，**未被选中的 Future 会被自动 drop**，从而触发取消。
#### 原理
- `select` 同时 poll 多个 Future
- 一旦某个分支完成：
    - 其他 Future 被 drop
    - 等价于 **Drop-based cancellation**
#### 示例

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    tokio::select! {
        _ = sleep(Duration::from_secs(1)) => {
            println!("fast path");
        }
        _ = async {
            sleep(Duration::from_secs(10)).await;
            println!("slow path");
        } => {}
    }
}
```

#### 行为特征

- ❌ 被取消 Future 不会继续执行
- ❌ 不保证清理
- ✅ 语义简洁
- ✅ 编译期结构清晰
    

####  使用场景

- 超时控制
- 多路等待
- 并发竞速（race）
    

---

## 总结



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
