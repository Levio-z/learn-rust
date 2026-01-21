---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
- [1. 定义错误类型](Rust-crate-thiserror-实现一个error需要哪些工作，Error做了哪些简化.md#1.%20定义错误类型)
- [2. 展示友好：实现 Display trait和Debug（派生）](#2.%20展示友好：实现%20Display%20trait和Debug（派生）)
- [3. 与标准错误兼容和返回底层错误：实现 std::error::Error trait以及最好实现source，返回底层错误](#3.%20与标准错误兼容和返回底层错误：实现%20std%20error%20Error%20trait以及最好实现source，返回底层错误)
- [4. From：从其它错误类型转换（可选，但常用）](#4.%20From：从其它错误类型转换（可选，但常用）)



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 1. 定义错误类型

- 可以是枚举（enum）或结构体（struct），**枚举更常用，因为它能表示多种错误情况**。
- 每个错误变体（variant）可以携带额外信息。
```rust
#[derive(Debug)]
pub enum MyError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Msg(String),
}
```

### 2. 展示友好：实现 `Display` trait和Debug（派生）
**让错误能以用户友好的方式格式化输出，也是实现error Trai的必要步骤**
```rust
#[derive(Debug)]
pub enum MyError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Msg(String),
}

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
### 3. 与标准错误兼容和返回底层错误：实现 `std::error::Error` trait以及最好实现source，返回底层错误
- 使错误类型能与标准错误处理体系兼容。
- 通常需要实现 `source()` 方法，返回导致当前错误的底层错误（可选，推荐实现）。
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
- [Rust-crate-thiserror-为什么要实现Error](Rust-crate-thiserror-为什么要实现Error.md)

### 4. From：从其它错误类型转换（可选，但常用）
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
- 实现 `Display`和派生Debug
- 实现 `Error`，**标准错误**
- 实现 `From`（根据需求）和source

### `thiserror::Error` 做了哪些简化？

`thiserror` 是一个非常流行的 Rust 宏库，专门用来简化错误类型的定义，实现了对上述所有繁琐步骤的自动生成。
```rust
use thiserror::Error;
#[derive(Debug, Error)]

enum ParserError {

    // Use #[from] to automatically implement From<io::Error> and implicitly set #[source]

    #[error("Failed to read file: {0}")]

    IoError(#[from] io::Error),

  

    // Use #[from] to automatically implement From<serde_json::Error> and implicitly set #[source]

    #[error("Failed to parse JSON")]

    JsonError(#[from] serde_json::Error),

  

    // Manually specify #[source], do not automatically implement From trait

    #[error("Data validation failed")]

    InvalidData {

        #[source] // Explicitly mark the source error

        cause: InvalidDataError,

    },

}

  

// Custom business logic error

  

#[derive(Debug, Error)]

#[error("Invalid data: {0}")]

struct InvalidDataError(String);
se thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
	//Display
    #[error("IO error: {0}")]

    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),

    #[error("Custom error: {0}")]
    Msg(String),
}
```
-  `#[derive(Debug)]`:实现Debug
- `#[error("Invalid data: {0}")]`:实现Display
- `#[derive(Error)]`：实现Error
- `#[from]`自动实现from和source
	-  `#[source]`:建立错误链，但拒绝自动转换

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
