### 背景
- **为你的 Rust 工具链添加对 RISC-V 架构（64位、无操作系统）的交叉编译支持**。
    - 编译出来的程序可以在 QEMU（或实际 RISC-V 开发板）上运行
    - Rust 默认只安装了你当前主机的编译目标（比如x86_64-unknown-linux-gnu）。而交叉编译目标（如 ARM、RISC-V 等）必须手动添加。
### Step1:添加支持
```
rustup target add riscv64gc-unknown-none-elf
```

#### 详解
> “请帮我安装一个新的编译目标，目标平台是 `riscv64gc-unknown-none-elf`，这样我可以把 Rust 项目编译成能在这个平台运行的二进制。”
> 
> 各字段含义：`riscv64gc-unknown-none-elf`

|             |                                              |
| ----------- | -------------------------------------------- |
| 字段          | 含义说明                                         |
| `riscv64gc` | 架构：RISC-V 64位，支持 **G** 扩展（IMAFD）和 **C** 压缩指令 |
| `unknown`   | 厂商未知（通用）                                     |
| `none`      | 没有操作系统（裸机，bare-metal）                        |
| `elf`       | 目标输出格式是 ELF（Executable and Linkable Format）  |
### Step2:添加build设置
然后在 os目录下新建 .cargo目录，并在这个目录下创建 config文件，并在里面输入如下内容：
```rust
# os/.cargo/config 
[build] 
target = "riscv64gc-unknown-none-elf"
```
> 它告诉 Cargo：**所有构建默认使用的目标平台是 `riscv64gc-unknown-none-elf`。**
> 即：
> > “我这个项目主要是写给 RISC-V 64 GC 裸机平台的，默认就不要给我编译成 x86_64 Linux 了。”

[交叉编译基本概念](../../../../../../../../Projects/开源操作系统训练营/2025春夏/第二阶段：rcore/rCore-Turial-note/概念合集/交叉编译/交叉编译基本概念.md)

当然，这只是使得我们之后在 cargo build的时候不必再加上 --target参数的一个小 trick。如果我们现在执行 cargo build，还是会和上一小节一样出现找不到标准库 std 的错误。于是我们需要在着手移除标准库的过程中一步一步地解决这些错误。