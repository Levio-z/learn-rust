---
tags:
  - permanent
---
## 1. 核心观点  

`futures：：select` 宏可以同时运行多个未来，允许用户在任何 future 完成后立即响应。'
，因为`select`中使用的未来必须实现两个 `Unpin` 以及`（FusedFuture`）特性。

```rust
use futures::{future, select};

async fn count() {
    let mut a_fut = future::ready(4);
    let mut b_fut = future::ready(6);
    let mut total = 0;

    loop {
        select! {
            a = a_fut => total += a,
            b = b_fut => total += b,
            complete => break,
            default => unreachable!(), // never runs (futures are ready, then complete)
        };
    }
    assert_eq!(total, 10);
}

```
同样，`FusedFuture` 特质是必需的，因为 `select` 不能在未来完成后轮询。`FusedFuture` 由追踪是否完成的未来实现。这使得可以在循环中使用 `select`，只轮询尚未完成的未来。这可以在上面的例子中看到，`a_fut` 或 `b_fut` 已经完成了第二次循环。因为 `future：：ready` 实现所返回的未来 `FusedFuture`，**它能告诉 `Select` 不要再轮询它。**
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### fuse
`fuse()` 是 `futures::FutureExt` 提供的一个**适配器方法**，用于把一个 `Future` 包装成 **`FusedFuture`**。  
其核心语义是：**一旦 Future 完成（Ready），后续再被轮询时将永远返回 `Pending`，且不会再产生任何副作用**。

---

### 初始设置：`fuse()` 解决的问题

在 `select!` / `select_biased!` 等**多路竞争（race）**场景中：

- `select!` 可能在同一个 `poll` 周期或后续调度中 **再次轮询已经完成的 Future**
- **普通 Future**：完成后再次 `poll` → **行为未定义 / panic**
- **`FusedFuture`**：完成后再次 `poll` → **稳定返回 `Pending`**

👉 `fuse()` 的作用就是 **把“只能完成一次的 Future”变成“完成后安全失效的 Future”**

### 初始设置：源码层面的本质（抽象视角）

`fuse()` 内部本质是一个状态机封装：

`enum State {     Pending(F),     Done, }`

- 第一次 `poll`：
    
    - 若内部 `F` 返回 `Ready` → 状态切到 `Done`
        
- 之后的 `poll`：
    
    - 直接返回 `Poll::Pending`
        

并同时实现：

`trait FusedFuture {     fn is_terminated(&self) -> bool; }`

用于让 `select!` 判断该 Future 是否已经“失效”。


### 初始设置：为什么 `select!` **强制要求** `fuse()`

`futures::select!` 的设计约束是：

> **参与竞争的 Future 必须是可安全重复 poll 的**

原因：

- `select!` 是一个宏，会生成一个 **状态机**
- 在不同分支未命中时，**未完成的 Future 会被再次 poll**
- 若某个分支先完成，其他分支可能仍被 poll（用于状态推进）
    

因此：

- ❌ 未 `fuse` 的 Future：完成后再 poll → UB / panic
    
- ✅ `fuse` 后的 Future：完成后再 poll → 安全 `Pending`
### 初始设置：与 Tokio `select!` 的对比（重要）

| 框架                 | 是否需要 `fuse()` | 原因                  |
| ------------------ | ------------- | ------------------- |
| `futures::select!` | ✅ 必须          | 依赖 `FusedFuture` 语义 |
| `tokio::select!`   | ❌ 不需要         | Tokio 在宏内部自动处理完成态   |


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-fuse-terminated](Rust-future-fuse-terminated.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
