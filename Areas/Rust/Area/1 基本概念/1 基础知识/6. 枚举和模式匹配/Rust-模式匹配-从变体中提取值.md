---
tags:
  - permanent
---
## 1. 核心观点  

匹配臂的另一个有用功能是它们可以绑定到与模式匹配的值部分。这就是我们从枚举变体中提取值的方法。
## 2. 背景/出处  
- 来源：https://rust-book.cs.brown.edu/ch06-02-match.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
```rust
#[derive(Debug)] // so we can inspect the state in a minute
enum UsState {
    Alabama,
    Alaska,
    // --snip--
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}
fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {state:?}!");
            25
        }
    }
}


```
- **此时，`state` 的绑定将是值 `UsState：：Alaska`**。
	- 然后，我们可以在 `println！` 表达式中使用该绑定，从而从 `Quarter` 的 `Coin` 枚举变体中获取内部状态值。

### 简单示例2：Option
```rust
    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            None => None,
            Some(i) => Some(i + 1),
        }
    }

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);

```
将变量绑定到里面的数据，然后基于它执行代码


### 语法
接下来是`match`臂。臂有两部分：模式和一些代码。
- 模式：这里的第一个臂有一个模式，即值 `Coin：:P enny`，=
- 分隔符：然后是 `=>` 运算符，用于分隔模式和要运行的代码。
- 代码：本例中的代码 只是值 `1`。每个臂都用逗号与下一个臂分隔。
	- 与每个臂关联的代码是一个表达式，匹配臂中表达式的结果值是为整个`匹配`表达式返回的值。
	- 如果要在匹配臂中运行多行代码，则必须使用大括号，并且臂后面的逗号是可选的。

语法逻辑
- 当`匹配`表达式执行时，它会按顺序将结果值与每个臂的模式进行比较。**如果模式与值匹配，则执行与该模式关联的代码。如果该模式与值不匹配，则执行将继续到下一臂**，就像在硬币分类机中一样。我们可以根据需要拥有任意数量的手臂：在示例 6-3 中，我们的`匹配`有四个手臂。



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- 
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 分类的原子笔记
- [ ] 实现的原子笔记
