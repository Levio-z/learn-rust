---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

### Ⅱ. 应用层

### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 引入依赖
```
[dependencies]
tokio = { version = "1", features = ["full"] }
```
### 在 `main` 函数上使用 `tokio::main` 注解
```
#[tokio::main]
async fn main() { ... }
```

`#[tokio::main]` ` 注解会初始化 Tokio 运行时，并启动一个异步任务来运行 ` `main` 中的代码。本指南稍后将更详细地解释该注解的作用，以及如何在不使用注解的情况下使用异步代码（这将为您提供更大的灵活性）。
### 基本例子
```rust
// Define an async function.
async fn say_hello() {
    println!("hello, world!");
}

#[tokio::main] // Boilerplate which lets us write `async fn main`, we'll explain it later.
async fn main() {
    // Call an async function and await its result.
    say_hello().await;
}
```
- 初始化 Tokio 运行时环境并创建一个初始任务来运行异步 `main` 函数的。
- `say_hello` 是一个异步函数，调用它时，必须在调用后加上 `.await` 才能将其作为当前任务的一部分运行。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
