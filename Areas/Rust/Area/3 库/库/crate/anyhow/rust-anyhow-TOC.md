`anyhow` 是 Rust 中专为 **应用层错误处理** 设计的库。
- 简化错误定义、传递、封装；  
- 让开发者无需为每个错误定义复杂类型；  
- 提供漂亮的错误链打印和调试输出。
### 使用场景
- `anyhow` 依赖 Rust 的标准库（`std`），而多数嵌入式环境使用的是 `no_std`，即不带标准库的环境。
### **核心特点**

| 特点                               | 说明                                                 |
| -------------------------------- | -------------------------------------------------- |
| `anyhow::Error`                  | 一个胖胖的、类型擦除的动态错误容器（`Box<dyn Error + Send + Sync>`）。 |
| `anyhow::Result<T>`              | 实际是 `Result<T, anyhow::Error>`，用于统一错误返回类型。         |
| `.context()` / `.with_context()` | 给错误加上下文信息，形成详细的错误链。                                |
| 自动捕获链式 `source()`                | 能打印出多层错误原因，而不仅仅是最表层的描述。                            |
#### 常见用法

#### 类型 anyhow::Result
```
use anyhow::Result;

fn main() -> Result<()> {
    Ok(())
}
```
- 等价于 `Result<T, anyhow::Error>`
- 避免每个函数写出复杂的错误类型。
- 适合顶层 `main`、业务逻辑函数。
#### 快速错误构造
- **定义**：`bail!` 是一个宏，用来**立即返回一个错误**。
- **等价写法**：
    `return Err(anyhow!("message"));`
- 用法更简洁，特别适合函数早退出。

```rust
use anyhow::{anyhow, Result};

fn do_work() -> Result<()> {
    Err(anyhow!("something went wrong: {}", 42))
}
```
#### 自动错误转换
```rust
use anyhow::Result;
use std::fs;

fn read_file(path: &str) -> Result<String> {
    let content = fs::read_to_string(path)?; // 自动转成 anyhow::Error
    Ok(content)
}
```
- 自动将各种错误转为 `anyhow::Error`，免写 `map_err`。
- 这是 `anyhow` 最大的省力点。

#### 上下文信息：`Context` trait

```rust
use anyhow::{Context, Result};
use std::fs;

fn load_config() -> Result<String> {
    let content = fs::read_to_string("config.toml")
        .context("failed to read config.toml")?;
    Ok(content)
}
```
- `.context("...")` 给错误加额外说明，方便调试。
- 推荐在边界（I/O、网络、解析）添加。
#### with_context
`with_context` 是 **`anyhow::Context` trait** 提供的方法，用于给错误添加额外上下文信息，提升调试和定位问题的能力。它通常与 `?` 操作符配合使用，使错误链更加可读。
**延迟生成上下文**  
因为 `context` 是闭包，只有在错误发生时才会调用，提高性能（避免不必要的字符串构建）。
**避免过早生成复杂字符串，尤其在高性能或大量 I/O 场景下**

#### 错误链查看：`.root_cause()` / `Debug`

```rust
fn main() -> Result<()> {
    if let Err(err) = load_config() {
        eprintln!("Error: {:?}", err);          // 打印错误链
        eprintln!("Cause: {}", err.root_cause()); // 最底层原因
    }
    Ok(())
}
```
生产调试中常用。
.chain() 可遍历所有错误层。

#### ensure!
`ensure!(cond, "msg")` 等价于：
```rust
if !cond {
    bail!("msg");
}
```

### 文档
- Readme：https://crates.io/crates/anyhow
### 原理
[anyhow ？传播错误原理](原理/anyhow%20？传播错误原理.md)
[anyhow context](原理/anyhow%20context.md)