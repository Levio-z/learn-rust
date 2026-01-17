
```rust

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

`#[error("...")]` 
- **自动生成 `Display` 实现**  
- **必须有**：通过 `#[error("...")]` 属性宏，自动根据字符串模板格式化错误消息。
	- 模板化可以使用0，引入底层错误
在带有 `#[from]` 的字段上
- 生成 `From` 实现
	- 自动生成从对应错误类型的转换实现，方便使用 `?`。
- **隐式**实现 `source()` 
	- 当你写了 `#[from]`，`thiserror` 会自动为你生成 `From` 实现，同时 **隐式** 认为这个字段是底层错误，自动帮你实现 `source()` 返回该字段的引用。
显示使用`#[source]`
- 但你可以通过显式的 `#[source]` 标记来告诉宏，哪个字段是“底层错误”用以实现 `source()`，即使它没有 `#[from]`。
- 反之，字段可能只标记了 `#[source]`，不生成 `From` 实现，这种情况常用于你只想实现错误链，而不想自动转换。