`thiserror` 是一个 Rust 宏库，提供方便的 **派生宏（derive macro）** 用于自动为自定义错误类型实现标准库中的 `std::error::Error` trait。
换句话说，它帮助开发者快速定义错误类型，而不用手写 `impl Error`、`Display` 和 `Debug`。
[Error](../../../复用概念/std/Error.md)
### 主要作用
换句话说，它帮助开发者快速定义错误类型，而不用手写 `impl Error`、`Display` 和 `Debug`。
手写这些实现不仅繁琐，而且容易出错，尤其是当错误枚举类型有很多变体时。  
`thiserror` 通过 derive 宏，把这些重复工作自动生成，大大减少了模板化代码（boilerplate）。
### Details  细节
- 此错误故意不出现在您的公共 API 中。你得到的东西就像你手工编写了一个 `std：：error：：Error 的实现一样` ，从手写 impls 切换到 thiserror 或反之亦然并不是一个突破性的变化。
- 错误可能是枚举、带命名字段的结构、元组结构或单元结构。
- 如果您提供 `#[error（“. ")]` 结构或枚举的每个变体上的消息，如上面的 example.
	- 这些消息支持从错误中插入字段的简写。
		- `#[error("{var}")]` ⟶ `write!("{}", self.var)`
		- `#[error("{0}")]` ⟶ `write!("{}", self.0)`
		- `#[error("{var:?}")]` ⟶ `write!("{:?}", self.var)`
		- `#[error("{0:?}")]` ⟶ `write!("{:?}", self.0)`
- 这些缩写可以与任何其他格式参数一起使用，这些格式参数可以是任意表达式。举例来说
### 例子
```rust
**use thiserror::Error;

use std::io;

  

/// 我们定义一个应用层的错误类型，涵盖两种情况：

/// - 文件 I/O 错误

/// - 配置解析错误

#[derive(Debug, Error)]

pub enum MyAppError {

    /// 自动为 io::Error 实现 From，允许用 `?` 自动转换。

    #[error("I/O error: {0}")]

    Io(#[from] io::Error),

  

    /// 配置文件格式有误

    #[error("Config parse error at line {line}: {msg}")]

    ConfigParse {

        line: usize,

        msg: String,

    },

}

fn read_file() -> Result<String, MyAppError> {

    let content = std::fs::read_to_string("config.toml")?; // io::Error → MyAppError::Io

    Ok(content)

}

  

fn parse_config(content: &str) -> Result<(), MyAppError> {

    if !content.starts_with("[config]") {

        return Err(MyAppError::ConfigParse {

            line: 1,

            msg: "Missing [config] section".to_string(),

        });

    }

    Ok(())

}

fn run_app() -> Result<(), MyAppError> {

    let content = read_file()?;

    parse_config(&content)?;

    println!("Config loaded successfully!");

    Ok(())

}

fn main() {

    if let Err(e) = run_app() {

        eprintln!("Application error: {}", e);

    }

}
```
分别展示了包装和自定义结构体实现

| 设计形式                     | 示例                                         | 特点                     |
| ------------------------ | ------------------------------------------ | ---------------------- |
| **包装型（tuple variant）**   | `Io(#[from] io::Error)`                    | 直接把一个已有类型包起来，起到“封装”作用。 |
| **结构体型（struct variant）** | `ConfigParse { line: usize, msg: String }` | 自己定义一组字段，携带更详细的上下文信息。  |

| 部分                           | 作用                                                 |
| ---------------------------- | -------------------------------------------------- |
| `#[derive(Debug, Error)]`    | 自动派生 `std::error::Error` 和 `std::fmt::Display` 实现。 |
| `#[error("I/O error: {0}")]` | 定义 `Display` 的格式化输出，`{0}` 是枚举变体里的 `io::Error`。     |
| `#[from]`                    | 告诉 `thiserror` 自动实现 `From<io::Error>`。             |
| `ConfigParse { line, msg }`  | 用结构体变体带上下文信息（行号、消息）。                               |
 debug模式：
	- `eprintln!("Application error: {}", e); -> eprintln!("Application error: {:?}", e); `
结果：
```rust
Application error: I/O error: 系统找不到指定的文件。
```

- debug
```rust
Application error: Io(Os { code: 2, kind: NotFound, message: "系统找不到指定的文件。" })
```
修改路径为Cargo.toml：
```rust
Application error: Config parse error at line 1: Missing [config] section
```
- debug
```rust
Application error: ConfigParse { line: 1, msg: "Missing [config] section" }
```




