---
tags:
  - permanent
---
## 1. 核心观点  

模式可用于_解构_[结构](https://doc.rust-lang.org/reference/items/structs.html)体、 [枚举](https://doc.rust-lang.org/reference/items/enumerations.html)和[元组](https://doc.rust-lang.org/reference/types/tuple.html) 。解构将值分解为其组成部分。使用的语法与创建此类值时几乎相同。

**当解构字段名与目标绑定变量名相同时，可以省略 `field_name: variable_name` 的写法**，直接写 `field_name`。


## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/patterns.html#r-patterns.ident.binding.mode-limitations-binding
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
```rust
match message {
    Message::Quit => println!("Quit"),
    Message::WriteString(write) => println!("{}", &write),
    Message::Move{ x, y: 0 } => println!("move {} horizontally", x),
    Message::Move{ .. } => println!("other move"),
    Message::ChangeColor { 0: red, 1: green, 2: _ } => {
        println!("color change, red: {}, green: {}", red, green);
    }
};

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
- [ ] 分类的原子笔记
- [ ] 实现的原子笔记
