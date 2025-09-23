- 使用 `Result<T、anyhow：：Error>` 或等效`的 anyhow：：Result<T>` 作为任何易出错函数的返回类型。
	- **`anyhow::Error` 是一个通用错误类型**，内部封装了任何实现了标准库 `std::error::Error` trait 的具体错误。
	- 直接使用 `anyhow` 提供的类型别名：
		- `type Result<T> = std::result::Result<T, anyhow::Error>;`
- 在函数中，使用 `？` 容易传播实现 [`std：：Error：：Error`](https://doc.rust-lang.org/std/error/trait.Error.html)trait.
	- [使用 `？` 容易传播实现的原因](#容易传播实现的原因)

#### 容易传播实现的原因
1. **`anyhow::Error`可以包装任何实现了标准库 `std::error::Error` trait的错误类型**
	- `anyhow::Error` 是一个**动态类型的胖错误包装器**，它内部是一个 `Box<dyn std::error::Error + Send + Sync + 'static>`。
	- 换句话说，它能包装**任何实现了标准库 `std::error::Error` trait**的错误类型。这样设计，让你在函数返回时可以“装箱”各种不同错误类型，避免了定义复杂的枚举错误类型。
2. **为什么 `?` 能轻松传播错误？**
	Rust 中 `?` 运算符是语法糖，等同于：
	```rust
	match expr {
    Ok(val) => val,
    Err(err) => return Err(From::from(err)),
	}
	```
3. `anyhow::Error` 如何实现自动转换？
	`anyhow::Error` 实现了：`From::from`,可以把std::error::Error转换为anyhow::Error
	```rust
	impl<E> From<E> for anyhow::Error where E: std::error::Error + Send + Sync + 'static

	```
- `?` 操作符在错误传播时，会隐式调用 `From::from`，实现自动类型转换。
举个例子：
```rust
fn foo() -> Result<(), MyError> {
    let _file = std::fs::File::open("foo.txt")?; // 这里 std::io::Error 会自动转换成 MyError
    Ok(())
}
```
等价于：
```rust
fn foo() -> Result<(), MyError> {
    let _file = match std::fs::File::open("foo.txt") {
        Ok(f) => f,
        Err(e) => return Err(MyError::from(e)),  // 这里隐式调用了 From::from
    };
    Ok(())
}

```

### 例子
```rust
use anyhow::{Result, Context};

  

fn read_file(path: &str) -> Result<String> {

    let content = std::fs::read_to_string(path)

        .with_context(|| format!("Failed to read file at path: {}", path))?;

    Ok(content)

}

  

fn main() -> Result<()> {

    let data = read_file("config.toml")?;

    println!("File content:\n{}", data);

    Ok(())

}
```
- anyhow::Result是anyhow提供的别名
	- `type Result<T> = std::result::Result<T, anyhow::Error>;`