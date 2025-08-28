`miette!` 宏非常像 Rust 的 `format!` 或 `anyhow!`，但更强大：
### Example
```rust
let x = 1;
let y = 2;
let report = miette!("{} + {} = {z}", x, y, z = x + y);

assert_eq!(report.to_string().as_str(), "1 + 2 = 3");

let z = x + y;
let report = miette!("{} + {} = {}", x, y, z);
assert_eq!(report.to_string().as_str(), "1 + 2 = 3");

```
#### 顺便声明诊断元信息
- 如严重性、错误码、提示、超链接、标注），而不仅仅是拼接一条字符串消息。
```rust
fn miette_macro() {
    let source = "(2 + 222222".to_string();
    let report = miette!(
        severity = Severity::Error,
        code = "expected::rparen",
        help = "always close your parens",
        labels = vec![LabeledSpan::at_offset(6, "here")],
        url = "https://example.com",
        "expected closing ')'"
    )
    .with_source_code(source);
    println!("{report:?}");

}

```
接受多个命名参数（diagnostic-like arguments）：

| 参数名      | 含义                                          |
| -------- | ------------------------------------------- |
| severity | 错误级别（如 `Error`、`Warning`、`Advice`）。         |
| code     | 错误代码，用于标识错误类型（通常是 `"模块::错误名"` 形式）。          |
| help     | 给用户的帮助提示（一般是下一步操作建议）。                       |
| labels   | `LabeledSpan` 列表，用于在源代码中高亮定位（类似 IDE 下划线提示）。 |
| url      | 错误详情文档的链接（点开后可以查看详细解释或教程）。                  |
| 最后字符串    | 主错误信息（最终输出的核心描述）。                           |
.with_source_code(source);
关联源代码，用于提供错误发生的上下文。
这样渲染出来的报错不仅是 “有个错”，还能标出：
- 哪一行
- 哪个位置（偏移量）    
- 哪个片段（span）
```rust
expected::rparen (link)

  × expected closing ')'
   ╭────
 1 │ (2 + 222222
   ·       ▲
   ·       ╰── here
   ╰────
  help: always close your parens
```
预期输出：
```rust
expected::rparen (link)

  × expected closing ')'
   ╭────
 1 │ (2 + 2
   ╰────
  help: always close your parens
```