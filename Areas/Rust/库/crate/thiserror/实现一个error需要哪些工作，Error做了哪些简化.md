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
