---
tags:
  - permanent
---
## 1. 核心观点  

- Rust 的模式匹配系统中，**绑定模式 (binding modes)** 用来决定匹配时变量是以「值」还是「引用」的方式绑定。
- 这一机制的目标是提高「人体工学」——让开发者**不必频繁地手动写 `ref` 或 `ref mut`**。

>为了让模式匹配更符合使用习惯（即“service better ergonomics”），Rust 在匹配时会自动调整绑定模式。当一个**引用类型的值**被[非引用模式](Rust-模式-模式分类-Non-reference%20patterns非引用模式.md)（non-reference pattern）**匹配时，Rust 会自动把这个匹配视为 `ref` 或 `ref mut` 绑定。


```rust
let x: &Option<i32> = &Some(3);
if let Some(y) = x {
    // y was converted to `ref y` and its type is &i32
}
```


## 2. 背景/出处  
- 来源：https://rust-book.cs.brown.edu/ch06-02-match.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 默认绑定模式
如果绑定模式没有显式具有 `ref`、`ref mut` 或 `mut`，则它使用_默认绑定模式_来确定变量的绑定方式。
默认绑定模式从使用移动语义的“移动”模式开始。




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-模式-模式分类-Non-reference patterns非引用模式](Rust-模式-模式分类-Non-reference%20patterns非引用模式.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 分类的原子笔记
- [ ] 实现的原子笔记
