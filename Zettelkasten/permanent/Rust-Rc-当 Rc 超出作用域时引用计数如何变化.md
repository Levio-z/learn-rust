---
tags:
  - permanent
---
## 1. 核心观点  

然后**每次调用 `clone` 时，计数都会增加 1**。当 `c` 超出范围时，计数会减少 1。我们不必像调用 `Rc：：clone` 来增加引用计数那样调用函数来减少引用计数：**当 `Rc<T>` 值超出范围时，`Drop` trait 的实现会自动减少引用计数**。


- [案例](https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=5171dbb87310291e18468d42864c0569)
## 2. 背景/出处  
- 来源：https://rust-book.cs.brown.edu/ch15-04-rc.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-Rc-基本概念](Rust-Rc-基本概念.md)
- 后续卡片：
	- 
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
