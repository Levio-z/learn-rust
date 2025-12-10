## 1. 核心观点  

`Option<T>` 是 Rust 标准库预导入（prelude）的一部分，用于表示一个值可能存在（`Some(T)`）或不存在（`None`）的情况。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 定义
```rust
enum Option<T> {
    Some(T),
    None,
}
```
它提供了一种类型安全的方式来处理“可空值”，避免传统语言中的空指针问题。由于被包含在前奏中，`Option`、`Some`、`None` 可以直接使用，无需显式导入。
### 类型推断与显式注解

- 当使用 `Some(value)` 时，Rust 可自动推断 `Option<T>` 的具体类型。
    
    `let some_number = Some(5);      // Option<i32> let some_char = Some('e');      // Option<char>`
    
- 当使用 `None` 时，编译器无法从字面值确定类型，因此必须显式指定：
    
    `let absent_number: Option<i32> = None;`




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-Option-与普通类型的区别](Rust-Option-与普通类型的区别.md)
	- [Rust-Option vs 空值-基本概念](../../../../../../1%20基础知识/RustBook/6.%20枚举和模式匹配/Rust-Option%20vs%20空值-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 分类的原子笔记
- [ ] 实现的原子笔记
