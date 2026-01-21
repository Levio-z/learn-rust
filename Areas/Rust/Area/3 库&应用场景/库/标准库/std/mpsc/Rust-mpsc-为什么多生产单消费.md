---
tags:
  - note
---
## 1. 核心观点  
### 初始设置：Rust `mpsc` 中的 fan-in 体现

```rust
let (tx, rx) = std::sync::mpsc::channel();

let tx1 = tx.clone();
let tx2 = tx.clone();
```

- `Sender` 可 clone → 多生产者
- `Receiver` 不可 clone → 单消费者

这在**类型系统层面**强制了 fan-in：[Rust-mpsc-fan-in-基本概念](fan-in/Rust-mpsc-fan-in-基本概念.md)

> **生产端可并发，消费端必须集中**

- `Sender<T>`：**可 Clone**
- `Receiver<T>`：**不可 Clone**

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 为什么说是“多生产者”
`let (tx, rx) = std::sync::mpsc::channel(); let tx2 = tx.clone();`
- 多个 `Sender<T>` 可以存在
- 多个线程可同时 `send`
- 内部通过队列 + 原子同步保证安全
### 初始设置：为什么“只能单消费者”
```rust
// ❌ Receiver 不能 clone
let rx2 = rx.clone(); // 编译错误
```
并且：
- `Receiver<T>` **不是 `Sync`**
- 不能被多个线程同时 `recv`
这是一个**有意为之的设计决策**：
> 防止消息被“竞争性消费”导致语义不清




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
