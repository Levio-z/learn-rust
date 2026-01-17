```
use std::fs::File;
use anyhow::Context; // anyhow 提供的 trait

let file = File::open("foo.txt")
    .context("failed to open foo.txt")?;
```
### 1. trait 扩展方法原理

- `anyhow::Context` 为 `Result<T, E>` 类型提供了额外的方法 `context` 和 `with_context`。
- Rust 的 trait 机制允许给已有类型（这里是 `Result<T, E>`）添加方法，只要在作用域里 `use` 了对应 trait。
- 所以当你写 `File::open(...).context(...)` 时，实际上是调用了 `Context` trait 的扩展方法。
    

源码大致：
```rust
pub trait Context<T> {
    fn context<C>(self, context: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;
}

impl<T, E> Context<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> Result<T> { ... }
}

```
- 作用对象：**所有 `Result<T, E>`，只要 `E` 是标准错误类型**
    
- 这里的 `context` 方法会：
    
    1. 检查 `self` 是否是 `Err(e)`
        
    2. 如果是 `Err(e)`，生成一个新的 `anyhow::Error`，把 `e` 和 `context` 包起来

    3. 返回 `Result<T, anyhow::Error>`

用伪代码表示：
```rust
fn context<C>(self, context: C) -> Result<T, anyhow::Error> {
    match self {
        Ok(v) => Ok(v),
        Err(e) => Err(anyhow::Error::new(e).context(context)),
    }
}

```

```
use std::fs::File;
use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let f = File::open("foo.txt")
        .context("failed to open foo.txt")?; // 如果失败，会附加 "failed to open foo.txt"
    Ok(())
}
```
清晰显示**上下文 + 原始错误**。
```
Error: failed to open foo.txt
Caused by: No such file or directory (os error 2)

```

### 2. 功能作用

- 给错误增加上下文信息，更容易调试。
    
- 如果 `File::open` 失败，它会生成一个新的错误，携带原始错误和你提供的描述字符串。
    
- 对于链式调用非常方便，例如：
```rust
let contents = File::open("foo.txt")
    .context("cannot open foo.txt")?
    .read_to_string(&mut String::new())
    .context("cannot read file")?;
```

### 3. 使用场景

1. **文件 I/O**：打开、读取、写入文件时，提供更明确的错误上下文。
    
2. **网络请求**：HTTP 请求失败时，加上 URL 或参数信息。
    
3. **配置解析**：解析失败时，加上配置文件路径和字段信息。