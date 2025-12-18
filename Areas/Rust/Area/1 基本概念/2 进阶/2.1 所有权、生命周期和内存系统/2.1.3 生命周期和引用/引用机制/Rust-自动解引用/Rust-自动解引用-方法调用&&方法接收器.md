---
tags:
  - permanent
---
## 1. 核心观点  

_方法调用_由一个表达式（ _接收器_ ）后跟一个点、一个表达式路径段和一个括号中的表达式列表组成。
参考
[Rust-方法调用-方法调用表达式](../../方法调用/Rust-方法调用-方法调用表达式.md)
### 定义与作用
- 方法接收器是 **定义在 `impl` 块中方法的第一个参数**，用于表示调用该方法的“对象实例”。
- 它决定了方法调用时如何传递所有权或引用。
- 语法形式为：
```rust
&self       // 共享借用接收器
&mut self   // 可变借用接收器
self        // 值接收器（移动所有权）
```
示例
```rust
impl Point {
    fn move_by(&mut self, dx: i32, dy: i32) { ... } // &mut self 是接收器
}
```
调用方式：
```rust
p.move_by(3, 4);   // 编译器自动展开为 Point::move_by(&mut p, 3, 4);
```
### 规则
- 方法调用会自动插入*，自动解引用：[1.0 Deref Trait 原理 - 方法调用自动插入 deref 链](1.0%20Deref%20Trait%20原理%20-%20方法调用自动插入%20deref%20链.md)
- 

## 2. 背景/出处  
- 来源：
	- [方法调用表达式](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

如果 `p` 是 `Box<Point>`，而方法需要 `&Point`，Rust 会自动调用：

`(&*p).move_by(3, 4);`

这就是为什么 `Box<String>`、`Rc<Vec<_>>` 等包装类型仍能直接调用其内部类型的方法。

https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=a8a86b8e706d0ccb64103bf6cd6cb3a5

*之前会调用，&String = &Str

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：
	- [Rust-方法调用-方法调用表达式](../../方法调用/Rust-方法调用-方法调用表达式.md)

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
