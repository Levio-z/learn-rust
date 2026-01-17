---
tags:
  - permanent
---
## 1. 核心观点  
- 基本定义
	- Rust 有一个非常强大的控制流结构，称为 `match`，它允许您**将一个值与一系列模式进行比较，然后根据匹配的模式执行代码**。
- 模式
	- 模式可以由文字值、变量名、通配符和许多其他东西组成; [第 19 章](https://rust-book.cs.brown.edu/ch19-00-patterns.html)涵盖了所有不同类型的模式及其作用。 
- 作用
	- `匹配`的力量来自模式的表现力以及**编译器确认所有可能的情况都已处理的事实**。
- vs if
	- 使用 `if`，条件需要计算为布尔值，**但这里它可以是任何类型**
- 语法:
	- [#语法 ](#语法 )
	- [匹配必须详尽,否则无法编译](#匹配必须详尽,否则无法编译)
## 2. 背景/出处  
- 来源：https://rust-book.cs.brown.edu/ch06-02-match.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

```
枚举和`匹配`表达式，其中枚举的变体作为其模式

使用 `if`，条件需要计算为布尔值，但这里它可以是任何类型。本例中的`硬币`类型是我们在第一行定义的`硬币`枚举。
### 语法
接下来是`match`臂。臂有两部分：模式和一些代码。
- 模式：这里的第一个臂有一个模式，即值 `Coin：:P enny`，=
- 分隔符：然后是 `=>` 运算符，用于分隔模式和要运行的代码。
- 代码：本例中的代码 只是值 `1`。每个臂都用逗号与下一个臂分隔。
	- 与每个臂关联的代码是一个表达式，匹配臂中表达式的结果值是为整个`匹配`表达式返回的值。
	- 如果要在匹配臂中运行多行代码，则必须使用大括号，并且臂后面的逗号是可选的。

语法逻辑
- 当`匹配`表达式执行时，它会按顺序将结果值与每个臂的模式进行比较。**如果模式与值匹配，则执行与该模式关联的代码。如果该模式与值不匹配，则执行将继续到下一臂**，就像在硬币分类机中一样。我们可以根据需要拥有任意数量的手臂：在示例 6-3 中，我们的`匹配`有四个手臂。
### 匹配必须详尽,否则无法编译
约束：Rust 中的匹配项是_详尽无遗_的：我们必须用尽每一个 为了使代码有效
编译报错提示：**Rust 知道我们没有涵盖所有可能的情况，甚至知道我们忘记了哪种模式**！
```
    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            Some(i) => Some(i + 1),
        }
    }

```
我们没有处理 `None` 情况，因此此代码将导致错误。幸运的是，这是一个 Rust 知道如何捕捉的错误。如果我们尝试编译此代码，我们将收到以下错误：
```rust
$ cargo run
   Compiling enums v0.1.0 (file:///projects/enums)
error[E0004]: non-exhaustive patterns: `None` not covered
 --> src/main.rs:3:15
  |
3 |         match x {
  |               ^ pattern `None` not covered
  |
note: `Option<i32>` defined here
 --> /rustc/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs:572:1
 ::: /rustc/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs:576:5
  |
  = note: not covered
  = note: the matched value is of type `Option<i32>`
help: ensure that all possible cases are being handled by adding a match arm with a wildcard pattern or an explicit pattern as shown
  |
4 ~             Some(i) => Some(i + 1),
5 ~             None => todo!(),
  |

For more information about this error, try `rustc --explain E0004`.
error: could not compile `enums` (bin "enums") due to 1 previous error

```
Rust 知道我们没有涵盖所有可能的情况，甚至知道我们忘记了哪种模式！Rust 中的匹配项是_详尽无遗_的：我们必须用尽每一个 为了使代码有效。[Rust-Option vs 空值-基本概念](Rust-Option%20vs%20空值-基本概念.md)





## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-模式匹配-匹配所有模式和 `_` 占位符](Rust-模式匹配-匹配所有模式和%20`_`%20占位符.md)
	- [Rust-模式匹配-从变体中提取值](Rust-模式匹配-从变体中提取值.md)
	- [Rust-模式-基本概念](Rust-模式-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

