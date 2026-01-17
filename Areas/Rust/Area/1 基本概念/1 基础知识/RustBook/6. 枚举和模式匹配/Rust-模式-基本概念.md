---
tags:
  - permanent
---
## 1. 核心观点  

模式用于将值与结构进行匹配，并可选择将变量绑定到这些结构内的值。它们还用于变量声明以及函数和闭包的参数。



## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/patterns.html#r-patterns.ident.binding.mode-limitations-binding
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
```rust
if let
    Person {
        car: Some(_),
        age: person_age @ 13..=19,
        name: ref person_name,
        ..
    } = person
{
    println!("{} has a car and is {} years old.", person_name, person_age);
}
```
### 模式作用
- [`let` declarations](https://doc.rust-lang.org/reference/statements.html#let-statements)
- [Function](https://doc.rust-lang.org/reference/items/functions.html) and [closure](https://doc.rust-lang.org/reference/expressions/closure-expr.html) parameters
- [`match` expressions](https://doc.rust-lang.org/reference/expressions/match-expr.html)
- [`if let` expressions](https://doc.rust-lang.org/reference/expressions/if-expr.html)
- [`while let` expressions](https://doc.rust-lang.org/reference/expressions/loop-expr.html#while-let-patterns)
-  [`for` expressions  `对于`表达式](https://doc.rust-lang.org/reference/expressions/loop-expr.html#iterator-loops)



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-模式-特点-可反驳](Rust-模式-特点-可反驳.md)
	- 功能
		- [Rust-模式-功能-解构](Rust-模式-功能-解构.md)
	- 模式分类
		- [Rust-模式-模式分类-Rest Pattern](Rust-模式-模式分类-Rest%20Pattern.md)
			- [Rust-模式-模式分类-StructPatternEtCetera](Rust-模式-模式分类-StructPatternEtCetera.md)
		- [Rust-模式-模式分类-Literal patterns 字面值模式](Rust-模式-模式分类-Literal%20patterns%20字面值模式.md)
		- [Rust-模式-模式分类-reference patterns引用模式](Rust-模式-模式分类-reference%20patterns引用模式.md)
- 相似主题：
	- [Rust-模式匹配-匹配所有模式和 `_` 占位符](Rust-模式匹配-匹配所有模式和%20`_`%20占位符.md)
	- [Rust-模式匹配-从变体中提取值](Rust-模式匹配-从变体中提取值.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

