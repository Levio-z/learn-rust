---
tags:
  - permanent
---
## 1. 核心观点

`async fn` 在编译期会被转换为**状态机类型**，递归 `async fn` 会导致状态机**自包含自身**，从而形成**无限大小类型**。  
要让递归异步函数合法，**必须引入一次间接层（indirection）**，典型方式是 `Box::pin`，将递归 Future 放到堆上。  
自 **Rust 1.77** 起，编译器已原生支持「**带分配间接的 async fn 递归**」，不再需要旧版本中“返回 boxed async block”的变通写法。

---

## 2. 背景 / 出处

- Rust async/await 的官方语义说明
- 编译器错误 `E0733: recursion in an async fn requires boxing`
- Rust 1.77 release：稳定支持 async fn 中的递归 + 堆分配间接
- futures-rs 中 `BoxFuture`、`FutureExt::boxed` 的历史用法
    

---

## 3. 展开说明

### 3.1 async fn 的本质：状态机展开

```rust
async fn foo() {
    step_one().await;
    step_two().await;
}
```

编译后等价于（概念上）：

```rust
enum Foo {
    First(StepOne),
    Second(StepTwo),
}
```

每一个 `.await` 对应状态机的一个变体，**Future 的大小在编译期是确定的**。

---

### 3.2 为什么递归 async fn 会失败

```rust
async fn recursive() {
    recursive().await;
    recursive().await;
}
```

概念等价展开：

```rust
enum Recursive {
    First(Recursive),
    Second(Recursive),
}
```

问题本质：

- `Recursive` **包含自身**
    
- 类型大小无法在编译期确定  
    ➡ **无限大小类型（infinitely-sized type）**
    

因此编译器拒绝，并给出：

```text
error[E0733]: recursion in an async fn requires boxing
```

---

### 3.3 解决思路：引入“间接层”（Indirection）

核心原则：

> **递归 Future 不能以内联方式嵌套，必须通过指针间接引用**

也就是把 Future 放到堆上，使状态机大小固定为一个指针大小。

---

### 3.4 Rust 1.77 之前的解决方案（历史方案）

由于编译器限制，**不能直接在 async fn 内 Box::pin 自身**，只能写成：

```rust
use futures::future::{BoxFuture, FutureExt};

fn recursive() -> BoxFuture<'static, ()> {
    async move {
        recursive().await;
        recursive().await;
    }
    .boxed()
}
```

特点：

- 外层是普通 `fn`
    
- 返回 `BoxFuture`
    
- 使用 `futures` crate
    
- 写法繁琐，但本质正确
    

---

### 3.5 Rust 1.77 之后：原生支持 async fn 递归

现在可以直接写：

```rust
async fn recursive_pinned() {
    Box::pin(recursive_pinned()).await;
    Box::pin(recursive_pinned()).await;
}
```

关键点：

- **async fn 本身仍然返回匿名 Future**
- 递归点使用 `Box::pin` 引入堆分配
- 编译器可以正确处理该模式
- 不再需要 `BoxFuture` / `.boxed()`

---

### 3.6 编译器层面的理解

- `Box::pin(fut)` → `Pin<Box<F>>`
- 状态机中只保存一个 **指针大小**
- 递归不再导致类型无限展开
- Pin 保证 Future 在堆上地址稳定，满足 async 的自引用需求
    

---

## 4. 与其他卡片的关联


## 5. 应用 / 启发

### 5.1 实际应用场景

- 异步 DFS / BFS
- 异步解释器 / AST 遍历
- 协议解析（递归下降）
- Actor / 状态机自循环
- 异步任务调度器内部逻辑
    

---

### 5.2 工程实践启发

- **一旦 async 出现递归，第一反应就是：是否需要 Box::pin**
    
- 性能敏感路径要评估：
    - 堆分配成本
    - 是否可以改写为循环 / 显式栈
- Rust 的 async 是**零成本抽象，但不是零分配语义**

---

### 5.3 思考与问题

- 是否所有递归 async 都值得存在？
- 能否用状态机 + loop 手写？
- 是否可以通过 trampoline / CPS 规避堆分配？
- async 递归与 Tokio task spawn 的语义差异？

---

## 6. 待办 / 进一步探索

- 对比 `async fn` 递归 vs 手写 enum 状态机
- 分析 `Box::pin` 在 LLVM 层面的代码生成
- 探索 async 递归在 no_std / embedded 环境下的可行性
- 研究 async + 生成器（Generator）未来 RFC 的可能演进
- 基准测试：递归 async vs 显式 loop 的性能差异