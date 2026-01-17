---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

一个普通的代码块（ `{ ... }` ）将源代码中的代码分组在一起，并为名称创建一个封装作用域。在运行时，代码块按顺序执行，并求值为其最后一个表达式的值（如果没有尾随表达式，则为单元类型（ `()` ））。

**与异步函数类似，异步代码块是常规代码块的延迟版本**。异步代码块将代码和名称绑定在一起，但在运行时不会立即执行，而是等待一个 Future 对象返回结果。要执行代码块并获取结果，必须使用 `await` 关键字。


### Ⅱ. 应用层

异步代码块是启动异步上下文并创建 Future 的最简单方法。**它通常用于创建仅在一个地方使用的小型 Future。**

通常情况下，你会更倾向于使用异步函数版本，因为它更简洁明了。然而，异**步代码块版本更加灵活，因为你可以在函数被调用时执行一些代码（通过将其写在异步代码块之外），并在等待结果时执行另一些代码（通过将其写在异步代码块之内）**。
### Ⅲ. 实现层

### **IV**.原理层

## 2. 背景/出处  
- 来源：[RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```
let s1 = {
    let a = 42;
    format!("The answer is {a}")
};

let s2 = async {
    let q = question().await;
    format!("The question is {q}")
};


```

如果我们执行这段代码， `s1` 将是一个可以打印的字符串，但 `s2` 将是一个 future； `question()` 不会被调用。要打印 `s2` ，我们首先需要 `s2.await` 。

### 类似函数，退出代码块必须用return
```rust
loop {
    {
        if ... {
            // ok
            continue;
        }
    }

    async {
        if ... {
            // not ok
            // continue;

            // ok - continues with the next execution of the `loop`, though note that if there was
            // code in the loop after the async block that would be executed.
            return;
        }
    }.await
}

```
要实现 `break` 你需要测试代码块的值（一种常见的做法是使用 [`ControlFlow`](https://doc.rust-lang.org/std/ops/enum.ControlFlow.html) 来获取代码块的值，这也允许使用 `?` ）。
### 异步代码块中的？
- 同样，在异步代码块中使用 `?` **会在遇到错误时终止 future 的执行，导致 `await` 代码块接收错误值**，但不会像在普通代码块中使用 `?` `
- 那样退出外层函数。你需要在 `await` 之后再使用一个 `?` ` 来实现这一点：

```
async {
    let x = foo()?;   // This `?` only exits the async block, not the surrounding function.
    consume(x);
    Ok(())
}.await?

```

令人恼火的是，这常常会让编译器感到困惑，因为（与函数不同）异步代码块的“返回”类型没有明确指定。您可能需要在变量上添加一些类型注解，或者使用 Turbofish 生成的类型才能使其正常工作，例如，在上面的示例中，使用 `Ok::<_, MyError>(())` 而不是 `Ok(())` 。

### 和异步函数

返回异步代码块的函数与异步函数非常相似。写成 `async fn foo() -> ... { ... }` 大致等价于 ` `fn foo() -> ... { async { ... } }` `。实际上，从调用者的角度来看，它们是等价的，从一种形式更改为另一种形式不会造成破坏性更改。此外，在实现异步特性时，你可以用另一种形式覆盖前者（见下文）。**但是，你必须调整类型，在异步代码块版本中显式地指定 ` `Future**` ： `async fn foo() -> Foo` 变为 ` `fn foo() -> impl Future<Output = Foo>` `（你可能还需要显式地指定其他边界，例如 `Send` 和 `'static` ）。



## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-异步编程-async-基本概念](Rust-异步编程-async-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
