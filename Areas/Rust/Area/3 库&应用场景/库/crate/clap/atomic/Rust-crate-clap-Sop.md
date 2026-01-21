---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层


### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 3.1 引入依赖
- 添加依赖：cargo add clap --features derive
- 导入
	- Parser：用于派生 CLI 解析器。
	- `Subcommand` ：用于派生子命令枚举，方便处理多层命令结构。
### 3.2  定义主 CLI 结构体
#### 2.1 定义结构体并使用`#[derive(Parser,Debug)]`宏
- 自动解析：通过`#[derive(Parser)]`宏自动为结构体实现Parser trait，derive宏会生成parse()方法，可将命令行参数转换为结构体实例
- `Debug`：实现调试输出能力，可以用 `{:?}` 格式打印结构体内容，方便调试。
#### 2.2 添加CLI的基本信息及参数﻿
```rust
#[command(version, about, long_about = None)]
```
- sop:填写about信息
##### 2.2.1 基本信息设置
>可从Cargo.toml自动获取name,version和about(description )信息

- version：自动从toml文件获取。
- `about` 用于短描述，填写about描述 （-h显示）
- `long_about` 可选长描述，这里设为 `None`。（--help显示）
##### 2.2.2 子命令设置
- 有子命令应该是个枚举 
```
#[clap(subcommand)]
pub sub: SubCommand,
```
- sop：确定name和about以及**嵌套结构**：子命令可以有自己的参数和子命令
- sop：架构提上添加`#[derive(Parser, Debug)]`
```rust
  #[command(name = "genpass", about = "Generate password")]
    GenPass(GenPassOpts),
    #[clap(subcommand, name = "base64", about = "Base64 encode or decode")]
    Base64(Base64SubCommand),
```
- [参考链接](https://github.com/learn-rust-projects/rcli/blob/e87a9f7786d3deba28b74e9651b97c562fb08183/src/cli/mod.rs)
##### 2.2.3 参数设置
- 参考信息：
	- [Rust-crate-clap-参数设计](参数设计/Rust-crate-clap-参数设计.md)
- 注释：` ///`后面的注释就是命令行help中显示的
- 提示该参数：value_name = "FILE"
```rust
#[arg(short, long, value_name="xxx",help = "Output format", default_value = "json",value_parser = parse_format)]
pub format: OutputFormat,
```
- sop：声明参数
	- 确定参数名称、类型、是否可选
- sop：short, long：短命令和长命令，确定是否有short冲突
- sop：value_name：指定显示名称
- sop：help：帮助信息
- sop：默认值：
	- default_value：提供一个**文本默认值**，后续再解析成目标类型
	- 直接提供一个 **类型安全的默认值**
- sop：value_parser = parse_format
	- [参考](https://github.com/Levio-z/rcli/blob/748c5161d30e5b6a593b5f7f65e5e936a2b5ec41/src/opts.rs)



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
