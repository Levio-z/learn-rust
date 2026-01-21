---
tags:
  - note
---

## 1. 核心观点  

**原则一句话版**：

> **当一个 `Future` 可能在多个线程之间被 `poll` 时，它必须实现 `Send`。**

更形式化地说：

- `Future: Send` ⇔ **Future 内部捕获的所有状态都是 `Send`**
- Rust 编译器会递归检查 async 状态机里保存的字段

### 关键
如果变量在 `.await` 之后还会用到 ⇒  
**它必须存进 Future ⇒ 必须是 Send**
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 什么时候会“突然报错说不是 Send”

1️⃣ async 中捕获了非 `Send` 类型
```rust
use std::rc::Rc;

async fn foo() {
    let x = Rc::new(1);
    bar().await;
    println!("{}", x);
}

```
此时：

- `x` 会被存进 Future 状态机
    
- `Rc<T>: !Send`
    
- 整个 Future ⇒ `!Send`
```

tokio::spawn(foo()); // ❌ 编译错误
```
#### 2️⃣ `await` 之前创建的变量会“跨 await 存活”

这是判断 Send 的**底层关键点**：
```
let x = Rc::new(1);
something().await;   // x 必须存进状态机
use(x);
```
如果变量在 `.await` 之后还会用到 ⇒  
**它必须存进 Future ⇒ 必须是 Send**



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  
