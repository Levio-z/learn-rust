---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层
- `await!` 是 Rust 编译器内置的一个特殊宏或操作符，用来“暂停”当前异步计算；
- 它接受一个实现了 `IntoFuture` trait 的表达式（通常是一个 `Future`），
- 作用是挂起当前异步任务，释放控制权，让执行环境（executor）可以切换到其他任务；
- 当该 Future 完成（返回 `Poll::Ready(value)`）时，`await!` 继续执行，并返回该值。
```rust
// future 是一个实现 Future<Output=usize> 的异步计算
let n = await!(future);
```
这里 `await!(future)` 等价于：轮询 `future`，如果 `Pending` 就让出控制，直到它变为 `Ready(n)`，然后 `n` 被赋值给 `let n`。
### Ⅱ. 应用层
### Ⅲ. 实现层
### **IV**.原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
