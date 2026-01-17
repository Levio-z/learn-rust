```rust
/*

You can derive a `Diagnostic` from any `std::error::Error` type.

  

`thiserror` is a great way to define them, and plays nicely with `miette`!

*/

use miette::{Diagnostic, NamedSource, SourceSpan};

use thiserror::Error;

  

#[derive(Error, Debug, Diagnostic)]
//   × oops!
#[error("oops!")]
#[diagnostic(
	// Error: oops::my::bad (link)
    code(oops::my::bad),
	// 文档链接自动指向 **docs.rs** 上托管的 Rust crate 文档
    url(docsrs),
	// help: try doing it better next time?
    help("try doing it better next time?")

)]

struct MyBad {

    // The Source that we're gonna be printing snippets out of.

    // This can be a String if you don't have or care about file names.

    #[source_code]
	// 封装源码及其名称的结构，通常用于诊断库（比如 `miette`）里表示错误源代码上下文。
    src: NamedSource<String>,

    // Snippets and highlights can be included in the diagnostic!

    #[label("This bit here")]

    bad_bit: SourceSpan,

}

  

/*

Now let's define a function!

  

Use this `Result` type (or its expanded version) as the return type

throughout your app (but NOT your libraries! Those should always return

concrete types!).

*/

use miette::Result;

fn this_fails() -> Result<()> {

    // You can use plain strings as a `Source`, or anything that implements

    // the one-method `Source` trait.
	
    let src = "source\n  text\n    here".to_string();

  

    Err(MyBad {

        src: NamedSource::new("bad_file.rs", src),

        bad_bit: (9, 4).into(),

    })?;

  

    Ok(())

}
```
- **`thiserror`**：Rust 中的派生宏库，用来方便定义自定义错误类型，自动实现 `std::error::Error`。
- **`miette`**：一个高级错误报告库，用于生成带有源码片段、高亮、提示、帮助信息的美观诊断报告。
- 在这个例子中，`thiserror` 定义错误，`miette` 提供用户友好的诊断输出。
### **导入部分**
```rust
use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;
```
- `miette::Diagnostic`：错误报告增强特性（代码、帮助、指向源码的标记）。
- `NamedSource`：表示有命名的源文件或文本。
- `SourceSpan`：表示在源文件中某个范围（偏移 + 长度）。
- `thiserror::Error`：定义错误类型的派生宏。
### **定义错误类型**
```rust
#[derive(Error, Debug, Diagnostic)]
#[error("oops!")]
#[diagnostic(
    code(oops::my::bad),
    url(docsrs),
    help("try doing it better next time?")
)]
struct MyBad {
    #[source_code]
    src: NamedSource<String>,
    #[label("This bit here")]
    bad_bit: SourceSpan,
}
```
`#[derive(Error, Debug, Diagnostic)]`
- `std::error::Error`（来自 `thiserror`）
- `Debug`（调试打印用）
- `Diagnostic`（来自 `miette`，让错误有“诊断信息”）
`#[error("oops!")]`
- 定义 `Display`，当错误被打印时显示 `"oops!"`。
`#[diagnostic(...)]`
- 提供给 `miette` 的元数据：
	- `code(oops::my::bad)` → 错误代码。
	- `url(docsrs)` → 链接到文档或帮助页
	- `help("try doing it better next time?")` → 显示的建议提示。
``#[source_code]``
- 指定哪个字段是源文件（用来显示片段）。
- 装源码及其名称的结构，通常用于诊断库（比如 `miette`）里表示错误源代码上下文。
- `NamedSource::new("bad_file.rs", src)` 创建一个带有文件名和内容的源代码块，方便后续错误定位和显示。
`#[label("This bit here")]`
- 给 `bad_bit` 标记的区域加标签。
- 这里 `(9, 4)` 可能代表源码中的某个位置或范围，比如起始字节位置 9，长度 4。
	- - `ffset: SourceOffset` —— 表示区间起始位置，通常是相对于整个源码的字节偏移或字符偏移。
	- `length: usize` —— 区间长度，表示该 span 覆盖的字节数或字符数。
### **定义函数**
```rust
use miette::Result;
fn this_fails() -> Result<()> {
    let src = "source\n  text\n    here".to_string();

    Err(MyBad {
        src: NamedSource::new("bad_file.rs", src),
        bad_bit: (9, 4).into(),
    })?;

    Ok(())
}

```
`miette::Result`
- 是 `Result<T, miette::Report>` 的别名，用来直接返回带诊断信息的结果。
错误触发
- 构造 `NamedSource`，表示文件 `"bad_file.rs"`，内容是三行字符串。
- `bad_bit: (9, 4).into()` → 9 是偏移，4 是长度，即在 `src` 的第 9~13 个字符处标记。
抛出错误
- `Err(MyBad { ... })?;` → 立即返回带错误的 `Result`。
### 主函数模拟
```rust
fn pretend_this_is_main() -> Result<()> {
    this_fails()?;
    Ok(())
}
```
直接调用 `this_fails`，如果出错，错误自动带上诊断信息，由 `miette` 报告。
### 结果
```
Error: oops::my::bad (link)

  × oops!
   ╭─[bad_file.rs:2:4]
 1 │     source
 2 │ ╭─▶   text
 3 │ ├─▶     here
   · ╰──── This bit here
   ╰────
  help: try doing it better next time?
  error: process didn't exit successfully: `target\debug\examples\miette.exe` (exit code: 1)
```