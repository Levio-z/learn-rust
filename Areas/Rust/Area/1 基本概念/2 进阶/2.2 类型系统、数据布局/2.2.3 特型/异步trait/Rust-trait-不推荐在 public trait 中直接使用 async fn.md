---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层
- 公共 trait 中使用 `async fn` 会触发 lint，因为**无法在签名中显式声明 `Future` 的 auto trait（如 `Send`）**
    
- 可以改写为 `fn -> impl Future + Send`，但这会形成**不可逆的 API 承诺**



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


**不建议在公共 trait 中使用 `async fn`，因为无法为其指定 auto trait 约束。**  
如果你只打算在自己的代码中使用该 trait，或者你并不关心 `Future` 是否实现了诸如 `Send` 这样的 auto trait，那么可以选择抑制这个 lint。  
`#[warn(async_fn_in_trait)]` 默认开启（rustc）。点击可查看完整的编译器诊断信息。

在 `mod.rs` 第 58 行第 31 列：  
你也可以把它等价地展开（desugar）为一个普通的 `fn`，返回 `impl Future`，并添加你需要的任何约束（例如 `Send`）；但需要注意的是，这些约束将来**不能被放宽**，否则就会构成一次**破坏性（breaking）的 API 变更**：  
`impl std::future::Future<Output = …> + Send`

---

**async**  
返回一个 `Future`，而不是阻塞当前线程。

在 `fn`、闭包或代码块前使用 `async`，可以把被标记的代码转换为一个 `Future`。因此，这段代码不会立即执行，而只有在返回的 `Future` 被 `.await` 时才会被求值。

我们已经编写了一本 async 书籍，详细介绍了 async/await 以及与使用线程相比的权衡取舍。


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-trait bound-基本概念](trait_bound/Rust-trait%20bound-基本概念.md)
	- [Rust-trait-rust提供的](../rust提供的trait/Rust-trait-rust提供的.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
