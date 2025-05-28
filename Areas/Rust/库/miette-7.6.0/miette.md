### 简介
`miette` 是 Rust 的诊断库。它包括一系列的 traits/protocols，允许你连接到它的 error reports工具，甚至编写你自己的 error reports！它允许您定义可以像这样打印出来的error类型（或以任何您喜欢的格式！）：
![](Pasted%20image%2020250528101920.png)
**注意：您必须启用 `“fancy”`crate 功能才能获得上面截图中的 fancy 报告输出。** 你应该只在你的顶级 crate 中这样做，因为这个奇特的特性会引入一些库等可能不需要的依赖项。
### 特征
- 通用[`诊断`](https://docs.rs/miette/latest/miette/trait.Diagnostic.html "trait miette::Diagnostic")协议，兼容（并依赖于） [`std：：error`](https://doc.rust-lang.org/nightly/core/error/trait.Error.html "trait core::error::Error")_std..
- 每个[`诊断程序`](https://docs.rs/miette/latest/miette/trait.Diagnostic.html "trait miette::Diagnostic")上的唯一错误代码。
- 自定义链接以获取有关错误代码的更多详细信息。
- 用于定义诊断元数据的超级方便的派生宏。
- 替换 [`anyhow`](https://docs.rs/anyhow)/[`eyre`](https://docs.rs/eyre) 类型[`result`](https://docs.rs/miette/latest/miette/type.Result.html "type miette::Result") ， [`reports`](https://docs.rs/miette/latest/miette/struct.Error.html "struct miette::Error")和 [`miette！`](https://docs.rs/miette/latest/miette/macro.miette.html "macro miette::miette") 宏，用于 `anyhow!`/`eyre!` 宏。
- 对代码片段数据的任意[`源代码的`](https://docs.rs/miette/latest/miette/trait.SourceCode.html "trait miette::SourceCode")通用支持，包括对`字符串的`默认支持。
`miette`crate 还捆绑了默认的 [`ReportHandler`](https://docs.rs/miette/latest/miette/trait.ReportHandler.html "trait miette::ReportHandler")，具有以下功能：
- 使用 ANSI/Unicode 文本的图形化[诊断输出](https://docs.rs/miette/latest/miette/#about)
- 支持单行和多行高亮显示
- 屏幕阅读器/盲文支持，门控 [`NO_COLOR`](http://no-color.org/)，和其他功能。
- 完全可定制的图形主题（或完全覆盖打印机）。
-  原因链印
- 将诊断代码转换为[支持端子](https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda)中的链接。
### 安装
```
 cargo add miette
```
fancy
```
cargo add miette --features fancy
```
### Example
[example 01](example%2001.md)
### 使用场景

| 场景            | 描述                                        |
| ------------- | ----------------------------------------- |
| CLI 应用程序错误报告  | 在终端中向用户展示漂亮、结构化的错误输出（例如，文件路径、行号、高亮片段、提示）。 |
| 配置文件/DSL/脚本解析 | 显示配置文件或脚本中出错位置的详细信息。                      |
| 编译器、解释器       | 高亮源代码错误（例如，Rust 编译器的“^”指向报错行的样式）。         |
| 教学和工具开发       | 帮助用户快速定位问题，提供额外的文档和帮助链接。                  |
### 使用最佳实践
- 类型
- 宏
	- [bail！](bail！.md)
	- [miette!(e)](miette!(e).md)