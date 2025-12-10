---
tags:
  - permanent
---
## 1. 核心观点  

`Unsize` 是 Rust 标准库里的**内在 trait（intrinsic trait）**，用于支持**动态大小类型（DST）转换**和**指针强制类型解析（coercion）**。它通常与 `CoerceUnsized` 一起工作，实现安全的类型强制。

## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/type-coercions.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
#### 1. 定义与作用

`pub trait Unsize<T> { }`

- **直译**：类型 `Self` 可以被“unsize”成 `T`
- **泛化理解**：`Self: Unsize<T>` 表示 **`Self` 可以安全地转换成 `T`**


- **应用场景**：
    1. **数组到切片**：`[T; n] -> [T]`
    2. **具体类型到 trait 对象**：`ConcreteType -> dyn Trait`
    3. **可变引用 / Box / Rc / Arc 指针转换**：`Box<T> -> Box<U>` 其中 `T: Unsize<U>`

> 注意：`Unsize` 是编译器内置 trait，用户不能直接实现，只能通过标准库类型（Box、Rc、&、&mut 等）使用。

|   |   |   |
|---|---|---|
|`Unsize`|判断 **具体类型可以被 unsize 为 DST**|`[T; n] -> [T]`, `ConcreteType -> dyn Trait`|

|                 |                                  |                                |
| --------------- | -------------------------------- | ------------------------------ |
| `CoerceUnsized` | 判断 **指针/引用可以强制转换成指向 DST 的指针/引用** | `Box<T> -> Box<U>`, `&T -> &U` |



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 原子笔记整理
	- [ ] 整理基本概念的笔记
	- [ ] String中的自动解引用，String的本质
	- [ ] 自动解引用规则
		- [ ] x
		- [ ] x
		- [ ] x
- [ ] 验证这个观点的边界条件  

