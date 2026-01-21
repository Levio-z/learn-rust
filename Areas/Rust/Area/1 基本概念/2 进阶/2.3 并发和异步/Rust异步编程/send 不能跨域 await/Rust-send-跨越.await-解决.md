---
tags:
  - permanent
---
## 1. 核心观点  

- [三种方案解决send问题](Rust-send-跨越.await-解决.md#三种方案解决send问题)
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
- 有些`async fn`状态机可以安全跨线程发送，而有些则不行。


如果我们把 `x` 存到变量里，它会在 `.await` 之后才被丢弃，这时`async fn` 可能已经在另一个线程上运行了。由于 `Rc` 不是`send`状态，允许它跨线程传输是不合理的。一个简单的解决办法是`drop` .wait 之前的 `Rc` 是 `wait，` 但遗憾的是，今天这功能已不工作。
### 封装所有非send变量的作用域
为了成功绕过这个问题，你可能需要引入一个封装所有非`send`变量的块作用域。这样会更容易 编译器判断这些变量不存在于 `await`点。
### 三种方案解决send问题
#### 使用可跨作用域数据结构
use std::sync::Arc;
### 防止跨越作用域
```
{
    let x = Rc::new(1);
    println!("{}", x);
}
bar().await;
```
### 使用本地线程spawn
```
tokio::task::spawn_local(async move {
    r1.do_something().await;
});
```
spawn_local 明确承诺：任务永远不会被移动到其他线程
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-fuse-terminated](Rust-future-fuse-terminated.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
