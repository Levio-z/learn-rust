`miette` 是 Rust 生态里一个专注于**优雅错误报告（pretty error reporting）**的库。它提供了比标准库 `Result` 更丰富、更带上下文信息的错误处理和显示。
而 `bail!` 宏正是 `miette`（以及 `anyhow`）中用来**快速提前返回错误**的工具。你可以把它理解为：
- `return Err(From::from(your_error));`
- 直接 `return Err(...)`，立刻终止当前函数
- `bail!` 的返回值是固定类型：`Err(miette::Report)`
### 使用场景
#### 提前退出
```rust
use miette::{bail, Result};

fn check_number(x: i32) -> Result<()> {
    if x < 0 {
        bail!("Negative number not allowed: {}", x);
    }
    Ok(())
}

```
等同于：
```rust
if x < 0 {
    return Err(miette!("Negative number not allowed: {}", x));
}
```
**带上下文的错误报告：**  
由于 `miette!` 宏内部会生成带上下文、堆栈信息的错误，`bail!` 可以直接利用这些优势，省去手动构造复杂错误类型。
