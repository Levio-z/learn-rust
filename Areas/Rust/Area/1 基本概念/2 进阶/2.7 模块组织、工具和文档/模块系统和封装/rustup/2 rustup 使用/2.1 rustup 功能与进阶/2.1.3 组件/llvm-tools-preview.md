### 定义
`llvm-tools-preview` 是 Rust 内置的一组 LLVM 工具的集合包，提供了一组用于**分析、调试、性能优化**的低层工具。
#### 包含的主要工具
| 工具名称                 | 功能说明                         |
| -------------------- | ---------------------------- |
| `llvm-objdump`       | 二进制反汇编（类似 GNU `objdump`）     |
| `llvm-nm`            | 查看目标文件符号表                    |
| `llvm-size`          | 查看目标文件各段大小                   |
| `llvm-profdata`      | Profile 数据合并，用于性能分析          |
| `llvm-cov`           | 覆盖率分析（与 `cargo-llvm-cov` 配合） |
| `llvm-ar`            | 管理 `.a` 静态库（类似 GNU `ar`）     |
| `llvm-strip`         | 去除符号表信息，减小二进制体积              |
| `llvm-as`/`llvm-dis` | LLVM IR 汇编/反汇编               |
| `llc`                | 将 LLVM IR 编译成目标汇编代码          |
#### 使用场景

|使用场景|说明|
|---|---|
|**代码覆盖率分析**|搭配 [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) 使用|
|**调试内核或裸机程序**|使用 `llvm-objdump` 反汇编分析|
|**生成 IR 分析优化路径**|使用 `llc` 查看优化前后的指令差异|
|**反向工程/嵌入式系统分析**|不依赖 GNU 工具链，全部基于 LLVM|