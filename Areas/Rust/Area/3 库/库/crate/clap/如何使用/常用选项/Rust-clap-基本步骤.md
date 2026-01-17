 定义命令基本内容思考模式
 
 - 基本步骤：
	- 添加依赖：cargo add clap --features derive
	- 导入Parser：use clap::Parser
	- 定义结构体并使用#[derive(Parser)]宏
	- 添加CLI信息：#[command(name，version, about, long_about = None)]
- 为数据结构添加Parser﻿
    - Parser特性：通过#[derive(Parser)]宏自动为结构体实现Parser trait
    - 自动解析：derive宏会生成parse()方法，可将命令行参数转换为结构体实例
- 添加CLI的基本信息及参数﻿
    - 基本信息：可从Cargo.toml自动获取name,version和about(description )信息
    - 参数类型：
        - 可选参数：使用Option`<T>`类型
        - 必选参数：直接使用具体类型
        - 标志参数：使用bool或计数类型
- 参数的可选性与short/long选项
	- short/long选项：
        - short：单字母参数，如-c
        - long：完整单词参数，如--config
	- 默认值：
        - default_value_t：直接指定字面量默认值
        - default_value：使用字符串并通过From trait转换,实际"asda0".into()
- 子命令的使用﻿
    - 子命令定义：使用#[derive(Subcommand)]宏修饰枚举
    - 嵌套结构：子命令可以有自己的参数和子命令
    - 自动转换：子命令会自动转换为小写形式作为命令名
- parse函数的实现与调用
	- 自动实现：通过derive宏自动生成parse函数
    - 使用方式：直接调用结构体::parse()解析命令行参数
    - 错误处理：参数不合法时会自动生成友好的错误信息
- 参数校验
	- 自己带的
	- 自定义
		- https://docs.rs/clap/4.5.53/clap/_derive/_tutorial/index.html#validated-values
- roData编译的时候字面量就会编译到程序里面，生命周期和程序一样长
### 初始模板
```rust
#[derive(Parser)]
#[command(
	name = "[]",
    version,
    about, 
    long_about,
)]
struct Cli {
    /// Optional name to operate on
    name: Option<String>,

    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}
```
### 版本（可选）
```rust
name = "myapp", //--version will show name
version ,       // version相关
```
- 名称必选，version默认和版本相关
### --help相关
#### 描述（可选）建议保留
先写一个简单即可
```
 about = "Short description here",
```
详细的
```
    about = "Short description here",
    long_about = "This is a longer description of your CLI tool.\nIt can span multiple lines, and provides more details in the help output."
```
- 默认就是help下面一行，保留这两个
