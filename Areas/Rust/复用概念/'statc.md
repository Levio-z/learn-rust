**疑问**：如果一个闭包满足 `Send + Sync`，是不是意味着它也自动 `'static` 呢？  
**简答**：**不是的。**`'static` 是一个**生命周期约束**，和 `Send + Sync` 这两个**线程安全 trait** 没有直接的从属关系，它们各自独立判断。
### 反例
```rust
fn main() {
    let name = String::from("Alice");

    let f = || {
        println!("Hello, {name}");
    };
}

```
- 这个闭包里捕获了一个局部变量 `name`（通过引用）。
- 它可能是 `Send` 和 `Sync`，但它**不是 `'static`**，因为它引用的是栈上的局部变量。
- 一旦 `main` 函数结束，`name` 被释放，再调用闭包就悬垂引用了。

> 所以 `'static` 是确保闭包**不含任何临时引用**的保证，是内存安全的核心保障。


```rust
let name = String::from("hello");

let f: Box<dyn Fn() + 'static> = Box::new(move || println!("{name}"));

```
- 这个闭包是 `'static`（它 move 了 `String` 进闭包中，闭包拥有数据所有权）。
- 但如果 `String` 里包含了 `Rc`、`RefCell` 或其他不实现 `Send + Sync` 的结构，  
    就会导致闭包不是线程安全的。

- `&T: Sync` 只要 `T: Sync`
- **`&T: Send` 只要 `T: Sync`**
