---
title: current.next.take() 的借用分析
tags:
  - permanent
---
## 1. 核心观点  
> 模式匹配访问一个字段时，Rust 会按层级自动借用它的父结构体，直到字段本身。**字段借用** = **字段本身的可变/不可变借用**。**父结构体借用** = **为了安全访问字段，Rust隐式借用父结构体（不可重叠）**。**借用层级**：嵌套字段 → 借用链条沿父结构体向上。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
- 例子
	- let ret_node = current.next.take(); 发生了几次借用，2次
	- 一次隐式可变**借用 `current`**（因为访问 current.next 必须先借用 current）
	    - `current` 是 `&mut ListNode`，这里会产生对 `current` 的可变借用
	    - 这一次借用的作用域是 **整个 take() 调用期间**
	- 一次可变借用**借用 `current.next`**
	    - `Option<T>` 的 `take()` 方法需要 `&mut self`
	    - `current.next` 的可变借用嵌套在 `current` 的可变借用之内
	    - 这个借用的作用域也是 **take() 调用期间**
	- **所有权移动**
	    - `take()` 内部 `mem::replace(self, None)` 把 `current.next` 的内容移动到返回值
	    - 移动完成后，可变借用结束
- while let Some(ref mut region) = current.next 
	- 再次借用current和current.next都会报错

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
