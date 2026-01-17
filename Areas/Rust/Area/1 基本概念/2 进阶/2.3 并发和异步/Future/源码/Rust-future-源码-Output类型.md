---
tags:
  - note
---
## 1. 核心观点  

- [关联类型](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#associated-types) `Output` 用于指定异步值的类型。例如, 上图中的 `async_read_file` 函数将返回一个 `Future` 实例，其 `Output` 被设置为 `File`
- [`poll`](https://doc.rust-lang.org/nightly/core/future/trait.Future.html#tymethod.poll) 方法可用于检查值是否已就绪。
- 它返回一个 [`Poll`](https://doc.rust-lang.org/nightly/core/future/trait.Future.html#tymethod.poll) 枚举，其定义如下：

```rust
pub enum Poll<T> {
    Ready(T),
    Pending,
}
```
- 当值已可用时（例如文件已从磁盘完全读取），它会被包装后返回 `Ready` 变体。否则返回 `Pending` 变体，向调用者表明该值尚不可用。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-future-基本概念-TOC](../Rust-future-基本概念-TOC.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
