[自定义目标配置清单](../../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.7%20模块组织、工具和文档/模块系统和封装/rustup/2%20rustup%20使用/2.3%20rustup%20常见问题与优化/2.3.1%20FAQ/自定义目标配置清单.md)
```json

{
    "llvm-target": "x86_64-unknown-none",
    "data-layout": "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
    "arch": "x86_64",
    "target-endian": "little",
    "target-pointer-width": "64",
    "target-c-int-width": "32",
    "os": "none",
    "executables": true
}
```
我们还需要添加下面与编译相关的配置项：
```json
"linker-flavor": "ld.lld",
"linker": "rust-lld",
```
- 指定链接器的参数风格，如 `ld.lld`
- **`linker`** 指定实际调用的链接器程序路径或名称。
	- `"rust-lld"` 是 Rust 工具链内置的 LLD 可执行文件名称，属于 Rust 自带的链接器包装器。
在这里，我们不使用平台默认提供的链接器，因为它可能不支持 Linux 目标系统。为了链接我们的内核，我们使用跨平台的 **LLD链接器**（[LLD linker](https://lld.llvm.org/)），它是和 Rust 一起打包发布的。
```json
"panic-strategy": "abort",
```
这个配置项的意思是，我们的编译目标不支持 panic 时的**栈展开**（[stack unwinding](https://www.bogotobogo.com/cplusplus/stackunwinding.php)），所以我们选择直接**在 panic 时中止**（abort on panic）。这和在 `Cargo.toml` 文件中添加 `panic = "abort"` 选项的作用是相同的，所以我们可以不在这里的配置清单中填写这一项。
```json
"disable-redzone": true,
```
我们正在编写一个内核，所以我们迟早要处理中断。要安全地实现这一点，我们必须禁用一个与**红区**（redzone）有关的栈指针优化：因为此时，这个优化可能会导致栈被破坏。如果需要更详细的资料，请查阅我们的一篇关于 [禁用红区](https://os.phil-opp.com/zh-CN/red-zone/) 的短文。
[23 红区](23%20红区.md)
```json
"features": "-mmx,-sse,+soft-float",
```

配置项被用来启用或禁用某个目标 **CPU 特征**（CPU feature）。通过在它们前面添加`-`号，我们将 `mmx` 和 `sse` 特征禁用；添加前缀`+`号，我们启用了 `soft-float` 特征。

`mmx` 和 `sse` 特征决定了是否支持**单指令多数据流**（[Single Instruction Multiple Data，SIMD](https://en.wikipedia.org/wiki/SIMD)）相关指令，这些指令常常能显著地提高程序层面的性能。然而，在内核中使用庞大的 SIMD 寄存器，可能会造成较大的性能影响：因为每次程序中断时，内核不得不储存整个庞大的 SIMD 寄存器以备恢复——这意味着，对每个硬件中断或系统调用，完整的 SIMD 状态必须存到主存中。由于 SIMD 状态可能相当大（512~1600 个字节），而中断可能时常发生，这些额外的存储与恢复操作可能显著地影响效率。为解决这个问题，我们对内核禁用 SIMD（但这不意味着禁用内核之上的应用程序的 SIMD 支持）。

禁用 SIMD 产生的一个问题是，`x86_64` 架构的浮点数指针运算默认依赖于 SIMD 寄存器。我们的解决方法是，启用 `soft-float` 特征，它将使用基于整数的软件功能，模拟浮点数指针运算。
```json
"rustc-abi": "x86-softfloat"
```
#### 完整配置清单
```json
{
    "llvm-target": "x86_64-unknown-none",
    "data-layout": "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
    "arch": "x86_64",
    "target-endian": "little",
    "target-pointer-width": "64",
    "target-c-int-width": "32",
    "os": "none",
    "executables": true,
    "linker-flavor": "ld.lld",
    "linker": "rust-lld",
    "panic-strategy": "abort",
    "disable-redzone": true,
    "features": "-mmx,-sse,+soft-float",
    "rustc-abi": "x86-softfloat"
}
```

