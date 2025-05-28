### 要实现 `std::error::Error`，你必须实现哪些 trait？为什么？这些 trait 分别起什么作用？
#### 核心要求
```rust
pub trait Error: Debug + Display {
    fn source(&self) -> Option<&(dyn Error + 'static)> { None }
    fn description(&self) -> &str { ... } // 已废弃
    fn cause(&self) -> Option<&dyn Error> { self.source() } // 已废弃
}
```
**必须实现**
- `Debug`
- `Display`
 **可选实现**
- `source()` → 如果你的错误有底层 cause/嵌套错误。
#### **为什么必须 Debug + Display？**

| Trait     | 用途                                |
| --------- | --------------------------------- |
| `Debug`   | 给开发者调试使用，用 `{:?}` 打印，机器可读（包含内部细节） |
| `Display` | 给用户展示用，用 `{}` 打印，人类友好（错误描述信息）     |
标准库和生态工具（比如 `Result`、`anyhow`、`miette`、`eyre`）都依赖：
- `Debug` → 递归打印整个错误链。
- `Display` → 给最终用户展示简明错误信息。
### 手动实现的基本骨架
```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct MyError;

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "something went wrong")
    }
}

impl Error for MyError {}

```
#### 如果你的错误包装了另一个错误：
```rust
#[derive(Debug)]
struct MyWrapperError {
    cause: std::io::Error,
}

impl fmt::Display for MyWrapperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "I/O wrapper error: {}", self.cause)
    }
}

impl Error for MyWrapperError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}

```
- `anyhow::Error` 能用 `.chain()` 展示多层错误链
- `?` 运算符自动传播时，底层错误信息也能被完整保留。
