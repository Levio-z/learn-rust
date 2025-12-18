现在我们可以在虚拟机中启动内核了。为了在[QEMU](https://www.qemu.org/) 中启动内核，我们使用下面的命令：
```bash
qemu-system-x86_64 -drive format=raw,file=target/x86_64-os/debug/bootimage-os-rust.bin
```

### 使用 `cargo run`
要让在 QEMU 中运行内核更轻松，我们可以设置在 cargo 配置文件中设置 `runner` 配置项：
```toml
# in .cargo/config.toml

[target.'cfg(target_os = "none")']
runner = "bootimage runner"
```
- 表示一旦你 `cargo run` 编译的是一个 `target_os = "none"` 的目标。构建完成后会自动调用 `bootimage runner` 并把可执行文件作为第一个参数传给它。[官方提供的 cargo 文档](https://doc.rust-lang.org/cargo/reference/config.html)讲述了更多的细节。
- 命令 `bootimage runner` 由 `bootimage` 包提供，参数格式经过特殊设计，可以用于 `runner` 命令。它将给定的可执行文件与项目的引导程序依赖项链接，然后在 QEMU 中启动它。`bootimage` 包的 [README文档](https://github.com/rust-osdev/bootimage) 提供了更多细节和可以传入的配置参数。
- 现在我们可以使用 `cargo run` 来编译内核并在 QEMU 中启动了。