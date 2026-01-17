---
tags:
  - permanent
---
## 1. 核心观点  
>内部字段的引用，它的**生命周期和借用**依赖于整个被匹配的表达式。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- 例子
```rust
while let Some(ref mut region) = current.next {
    // 使用 region
}
```
- region 生命周期 ≤ current.next 可变借用生命周期
- current.next 生命周期 ≤ current 的可变借用生命周期
```rust
current: &mut ListNode       // 父节点引用
current.next: &mut Option<Box<Node>>  // 隐式可变借用
region: &mut Node            // 对内部 Node 的借用
```
**current.next被借用，current被借用current.其他字段可以使用。**
模式匹配中**字段引用不再使用时，父字段的隐式借用也消失**
```rust
while let Some(ref mut region) = current.next {
    // 使用 region
    // region作用域结束
    // 可以重新借用&mut current
}
```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：[Rust-访问一个字段的方法时，隐式可变借用父结构体](../permanent/Rust-访问一个字段的方法时，隐式可变借用父结构体.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
