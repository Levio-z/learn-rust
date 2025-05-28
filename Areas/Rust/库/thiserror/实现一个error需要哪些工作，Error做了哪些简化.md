### 1. 定义错误类型

- 可以是枚举（enum）或结构体（struct），枚举更常用，因为它能表示多种错误情况。
- 每个错误变体（variant）可以携带额外信息。
```rust
#[derive(Debug)]
pub enum MyError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Msg(String),
}
```
### 2. 实现 `std::error::Error` trait
使错误类型能与标准错误处理体系兼容。
通常需要实现 `source()` 方法，返回导致当前错误的底层错误（可选，推荐实现）。
```rust
impl std::error::Error for MyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MyError::Io(e) => Some(e),
            MyError::Parse(e) => Some(e),
            MyError::Msg(_) => None,
        }
    }
}

```
### 3. 实现 `Display` trait
**让错误能以用户友好的方式格式化输出。**
```rust
impl std::fmt::Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyError::Io(e) => write!(f, "IO error: {}", e),
            MyError::Parse(e) => write!(f, "Parse error: {}", e),
            MyError::Msg(msg) => write!(f, "Error: {}", msg),
        }
    }
}
```
### 4. 从其它错误类型转换（可选，但常用）
通过实现 `From` trait，使错误类型可以从底层错误自动转换，方便 `?` 操作符使用。
```rust
impl From<std::io::Error> for MyError {
    fn from(err: std::io::Error) -> MyError {
        MyError::Io(err)
    }
}

impl From<std::num::ParseIntError> for MyError {
    fn from(err: std::num::ParseIntError) -> MyError {
        MyError::Parse(err)
    }
}

```
## 总结：完整实现一个错误类型至少要做

- 定义错误类型（enum/struct）
    
- 实现 `Display`
    
- 实现 `Error`
    
- 实现 `From`（根据需求）

### `thiserror::Error` 做了哪些简化？

`thiserror` 是一个非常流行的 Rust 宏库，专门用来简化错误类型的定义，实现了对上述所有繁琐步骤的自动生成。
```rust
se thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),

    #[error("Custom error: {0}")]
    Msg(String),
}
```
**自动生成 `Display` 实现**  
- **必须有**：通过 `#[error("...")]` 属性宏，自动根据字符串模板格式化错误消息。
- 自动实现 `std::error::Error` trait
	- 包括 `source()` 方法，自动关联底层错误。
- 自动生成 `From` 实现
	- 在带有 `#[from]` 的字段上，自动生成从对应错误类型的转换实现，方便使用 `?`。
		- 当你写了 `#[from]`，`thiserror` 会自动为你生成 `From` 实现，同时 **隐式** 认为这个字段是底层错误，自动帮你实现 `source()` 返回该字段的引用。
	- 但你可以通过显式的 `#[source]` 标记来告诉宏，哪个字段是“底层错误”用以实现 `source()`，即使它没有 `#[from]`。
	- 反之，字段可能只标记了 `#[source]`，不生成 `From` 实现，这种情况常用于你只想实现错误链，而不想自动转换。
```rust
#[derive(thiserror::Error, Debug)]
use std::io;

use serde_json;

use std::error::Error;

use thiserror::Error;

  

#[derive(Debug, Error)]

enum ParserError {

    // 使用 #[from] 自动实现 From<io::Error>，并隐式设置 #[source]

    #[error("Failed to read file")]

    IoError(#[from] io::Error),

  

    // 使用 #[from] 自动实现 From<serde_json::Error>，并隐式设置 #[source]

    #[error("Failed to parse JSON")]

    JsonError(#[from] serde_json::Error),

  

    // 手动指定 #[source]，不自动实现 From trait

    #[error("Data validation failed")]

    InvalidData {

        #[source]  // 显式标记 source 错误

        cause: InvalidDataError,

    },

}

  

// 自定义业务逻辑错误

#[derive(Debug, Error)]

#[error("Invalid data: {0}")]

struct InvalidDataError(String);

fn read_file(path: &str) -> Result<String, ParserError> {

    // ? 自动将 io::Error 转换成 ParserError::IoError

    let content = std::fs::read_to_string(path)?;

    Ok(content)

}

  

fn parse_json(json_str: &str) -> Result<serde_json::Value, ParserError> {

    // ? 自动将 serde_json::Error 转换成 ParserError::JsonError

    let data = serde_json::from_str(json_str)?;

    Ok(data)

}

fn validate_data(data: &str) -> Result<(), InvalidDataError> {

    if data.is_empty() {

        return Err(InvalidDataError("Data cannot be empty".into()));

    }

    Ok(())

}

  

fn process_data(data: &str) -> Result<(), ParserError> {

    validate_data(data).map_err(|e| ParserError::InvalidData { cause: e })?;

    Ok(())

}

fn main() {

    // 测试 IoError（自动 From 转换）

    let io_err = read_file("nonexistent.txt").unwrap_err();

    println!("IO Error: {:?}", io_err);

    println!("Source: {:?}", io_err.source()); // 可以获取底层 io::Error

  

    // 测试 InvalidData（手动 #[source] 包装）

    let validation_err = process_data("").unwrap_err();

    println!("Validation Error: {:?}", validation_err);

    println!("Source: {:?}", validation_err.source()); // 可以获取 InvalidDataError

}
```
结果：
```
IO Error: IoError(Os { code: 2, kind: NotFound, message: "系统找不到指定的文件。" })
Source: Some(Os { code: 2, kind: NotFound, message: "系统找不到指定的文件。" })
Validation Error: InvalidData { cause: InvalidDataError("Data cannot be empty") }
Source: Some(InvalidDataError("Data cannot be empty"))
```