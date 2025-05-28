- 这个 trait 是封闭的，不能在 miette 库外部为其它类型实现

### Example
```rust
use miette::{WrapErr, IntoDiagnostic, Result};
use std::{fs, path::PathBuf};

pub struct ImportantThing {
    path: PathBuf,
}

impl ImportantThing {
    pub fn detach(&mut self) -> Result<()> {...}
}

pub fn do_it(mut it: ImportantThing) -> Result<Vec<u8>> {
    it.detach().wrap_err("Failed to detach the important thing")?;

    let path = &it.path;
    let content = fs::read(path)
        .into_diagnostic()
        .wrap_err_with(|| format!(
            "Failed to read instrs from {}",
            path.display())
        )?;

    Ok(content)
}
```
打印时，最外层的错误将首先打印，下面将列举较低级别的潜在原因。
```rust
Error: Failed to read instrs from ./path/to/instrs.json

Caused by:
    No such file or directory (os error 2)
```
