---
tags:
  - permanent
---
## 1. 核心观点  

当一个模式有可能与它所匹配的值不匹配时，它被称为_可反驳_的模式。


## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/patterns.html#refutability
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
```rust
let (x, y) = (1, 2);               // "(x, y)" is an irrefutable pattern

if let (a, 3) = (1, 2) {           // "(a, 3)" is refutable, and will not match
    panic!("Shouldn't reach here");
} else if let (a, 4) = (3, 4) {    // "(a, 4)" is refutable, and will match
    println!("Matched ({}, 4)", a);
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
	- [Rust-模式匹配-匹配所有模式和 `_` 占位符](Rust-模式匹配-匹配所有模式和%20`_`%20占位符.md)
	- [Rust-模式匹配-从变体中提取值](Rust-模式匹配-从变体中提取值.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

