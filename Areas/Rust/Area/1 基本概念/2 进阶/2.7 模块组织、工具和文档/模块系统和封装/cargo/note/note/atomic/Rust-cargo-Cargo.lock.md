---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

`Cargo.lock` 包含有关您的依赖项的确切信息。它由 Cargo 维护，不应手动编辑。

- [1. 什么时候触发](Rust-cargo-Cargo.lock.md#1.%20什么时候触发)
- [2. 它出来的过程是怎样的？](Rust-cargo-Cargo.lock.md#2.%20它出来的过程是怎样的？)
- [3. 为什么需要这个新文件？](#3.%20为什么需要这个新文件？)
- [4. 我该拿它怎么办？](#4.%20我该拿它怎么办？)
- [案例](Rust-cargo-Cargo.lock.md#案例)

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 1. 什么时候触发
**第一次运行任何会导致 Cargo 尝试“解析”依赖的命令** 时自动生成。

最常见的情况是：当你执行 **`cargo build`** 或 **`cargo run`** 时，它就会悄然出现在你的项目根目录。

- **首次编译**：执行 `cargo build` 或 `cargo run`。
    
- **预检查**：执行 `cargo check`（即使不生成二进制文件，Cargo 也需要确定依赖版本）。
    
- **更新依赖**：当你运行 `cargo update` 时。
    
- **手动测试**：运行 `cargo test`。


### 2. 它出来的过程是怎样的？

你可以把这个过程想象成一次“从模糊到精确”的固定：

1. **读取意向**：Cargo 读取你的 `Cargo.toml`。假设你写了 `serde = "1.0"`，这代表“我想要 1.0 以上且不破环兼容性的最新版本”。
    
2. **解析版本**：Cargo 访问注册表（如 crates.io），发现当前最新的是 `1.0.150`。
    
3. **锁定结果**：Cargo 将 `1.0.150` 这个**确切的版本号**记录在 `Cargo.lock` 中。
    
4. **保存“快照”**：从此以后，除非你手动要求更新，否则无论以后 `serde` 出了 `1.0.151` 还是 `1.1.0`，你的项目构建时都会死死锁在 `1.0.150`。

### 3. 为什么需要这个新文件？

如果只有 `Cargo.toml`，可能会出现“今天代码能跑，明天就报错”的尴尬情况（因为第三方库更新了）。`Cargo.lock` 的存在是为了：

- **确定性**：确保在你的电脑上、你同事的电脑上、以及服务器上的构建结果**完全一致**。
    
- **效率**：下次构建时，Cargo 直接看 `Cargo.lock` 就知道该下载哪个版本，不需要重新计算版本冲突

### 4. 我该拿它怎么办？

- **不要手动修改它**：这个文件是给 Cargo 读的，不是给人读的。
    
- **是否提交到 Git？**
    
    - 如果你在写**可执行程序（Binary）**：**必须提交**。这样能保证所有用户用到的依赖版本和你开发时一模一样。
        
    - 如果你在写**库（Library）**：**通常不提交**（添加到 `.gitignore`）。因为你的库会被别人引用，别人应该根据他们项目的需求去锁定版本。
### 案例

```
[package] 
name = "hello_world" 
version = "0.1.0" 
[dependencies] 
regex = { git = "https://github.com/rust-lang/regex.git" }
```
- 您可以通过在 `Cargo.toml` 文件中定义一个特定的 `rev` 值来解决此问题，这样 Cargo 就可以准确地知道在构建软件包时要使用哪个版本：
```
[dependencies] regex = { git = "https://github.com/rust-lang/regex.git", rev = "9f9f693" }
```
引入 `Cargo.lock` 。有了它，您无需手动跟踪确切的版本：Cargo 会自动为您完成。当您有如下清单时：
```
[package]
name = "hello_world"
version = "0.1.0"

[dependencies]
regex = { git = "https://github.com/rust-lang/regex.git" }

```
Cargo 会获取最新的提交信息并将其写入您的系统中。 首次构建时会生成 `Cargo.lock` 。该文件内容如下所示：
```
[[package]]
name = "hello_world"
version = "0.1.0"
dependencies = [
 "regex 1.5.0 (git+https://github.com/rust-lang/regex.git#9f9f693768c584971a4d53bc3c586c33ed3a6831)",
]

[[package]]
name = "regex"
version = "1.5.0"
source = "git+https://github.com/rust-lang/regex.git#9f9f693768c584971a4d53bc3c586c33ed3a6831"

```
你可以看到这里有更多信息，包括确切的信息。 你之前用来构建的版本。现在当你把软件包交给其他人时， 即使你没有在配置中指定，它们也会使用完全相同的 SHA 值。 `Cargo.toml` 。




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
