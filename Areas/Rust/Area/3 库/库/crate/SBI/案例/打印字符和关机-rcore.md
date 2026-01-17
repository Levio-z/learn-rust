目前， SBI spec 已经发布了 v2.0-rc8 版本，但本教程基于 2023 年 3 月份发布的 [v1.0.0 版本](https://github.com/riscv-non-isa/riscv-sbi-doc/releases/download/v1.0.0/riscv-sbi.pdf) 。我们可以来看看里面约定了 SEE 要向 OS 内核提供哪些功能，并寻找我们本节所需的打印到屏幕和关机的接口。可以看到从 Chapter 4 开始，每一章包含了一个 SBI 拓展（Chapter 5 包含多个 Legacy Extension），代表一类功能接口，这有点像 RISC-V 指令集的 IMAFD 等拓展。每个 SBI 拓展还包含若干子功能。其中：


- Chapter 5 列出了若干 SBI 遗留接口，其中包括串口的写入（正是我们本节所需要的）和读取接口，分别位于 5.2 和 5.3 小节。在教程第九章我们自己实现串口外设驱动之前，与串口的交互都是通过这两个接口来进行的。顺带一提，第三章开始还会用到 5.1 小节介绍的 set timer 接口。
    
- Chapter 10 包含了若干系统重启相关的接口，我们本节所需的关机接口也在其中。


**内核应该如何调用 RustSBI 提供的服务呢？通过函数调用是行不通的，因为内核并没有和 RustSBI 链接到一起**，我们仅仅使用 RustSBI 构建后的可执行文件，因此内核无从得知 RustSBI 中的符号或地址。幸而， RustSBI 开源社区的 [sbi_rt](https://github.com/rustsbi/sbi-rt) 封装了调用 SBI 服务的接口，我们直接使用即可。首先，我们在 `Cargo.toml` 中引入 sbi_rt 依赖：

```rust
// os/Cargo.toml
[dependencies]
sbi-rt = { version = "0.0.2", features = ["legacy"] }
```

我们将内核与 RustSBI 通信的相关功能实现在子模块 `sbi` 中，因此我们需要在 `main.rs` 中加入 `mod sbi` 将该子模块加入我们的项目。在 `os/src/sbi.rs` 中，我们直接调用 sbi_rt 提供的接口来将输出字符：

```
// os/src/sbi.rs
pub fn console_putchar(c: usize) {
    #[allow(deprecated)]
    sbi_rt::legacy::console_putchar(c);
}
```
注意我们为了简单起见并未用到 `sbi_call` 的返回值，有兴趣的同学可以在 SBI spec 中查阅 SBI 服务返回值的含义。到这里，同学们可以试着在 `rust_main` 中调用 `console_putchar` 来在屏幕上输出 `OK` 。接着在 Qemu 上运行一下，我们便可看到由我们自己输出的第一条 log 了。

同样，我们再来实现关机功能：

```rust
// os/src/sbi.rs
pub fn shutdown(failure: bool) -> ! {
    use sbi_rt::{system_reset, NoReason, Shutdown, SystemFailure};
    if !failure {
        system_reset(Shutdown, NoReason);
    } else {
        system_reset(Shutdown, SystemFailure);
    }
    unreachable!()
}
```
这里的参数 failure表示系统是否正常退出，这会影响 Qemu 模拟器进程退出之后的返回值，我们则会依此判断系统的执行是否正常。更多内容可以参阅 SBI spec 的 Chapter 10。



> **sbi_rt 是如何调用 SBI 服务的**
> 
> SBI spec 的 Chapter 3 介绍了服务的调用方法：只需将要调用功能的拓展 ID 和功能 ID 分别放在 `a7` 和 `a6` 寄存器中，并按照 RISC-V 调用规范将参数放置在其他寄存器中，随后执行 `ecall` 指令即可。这会将控制权转交给 RustSBI 并由 RustSBI 来处理请求，处理完成后会将控制权交还给内核。返回值会被保存在 `a0` 和 `a1` 寄存器中。在本书的第二章中，我们会手动编写汇编代码来实现类似的过程。