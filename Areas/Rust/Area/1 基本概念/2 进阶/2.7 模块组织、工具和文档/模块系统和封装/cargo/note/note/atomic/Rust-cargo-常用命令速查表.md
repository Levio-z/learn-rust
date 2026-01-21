---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
- [创建](#创建)
- [编译](#编译)
- [编译+运行](#编译+运行)
- [编译启用优化功能](Rust-cargo-常用命令速查表.md#编译启用优化功能)
- [Cargo.lock](#Cargo.lock)
- [更新依赖](Rust-cargo-常用命令速查表.md#更新依赖)
- [添加依赖](#添加依赖)
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 创建
```rust
cargo new hello_world
```
- Cargo 默认使用 `--bin` 生成二进制程序。
- 要生成库文件，使用 `--lib` 参数。
- 禁用远程仓库：初始化一个新的 `git` 默认情况下使用仓库。如果您不希望这样做，请传递 `--vcs none` 。

### 编译
```rust
cargo build
```
Cargo 将获取新的依赖项及其所有依赖项，编译它们，并更新 `Cargo.lock` 文件
### 运行
```
./target/debug/hello_world
```
### 编译+运行
```
cargo run
```

`Cargo.lock` 新文件。它包含有关依赖项的信息。
### 发布编译启用优化功能
```
cargo build --release
```
- 开发时默认使用调试模式编译。由于编译器不进行优化，编译时间较短，但代码运行速度较慢。发布模式编译时间较长，但代码运行速度更快。


### `Cargo.lock`
- [Rust-cargo-Cargo.lock](Rust-cargo-Cargo.lock.md)

### 更新依赖
```
$ cargo update         # updates all dependencies
$ cargo update regex   # updates just “regex”

```
### 添加依赖

```
cargo add tokio --features rt,rt-multi-thread,macros,net
```
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
