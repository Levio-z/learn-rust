现在我们在 `_start` 函数结束后进入了一个死循环，所以每次执行完 `cargo test` 后我们都需要手动去关闭QEMU；但是我们还想在没有用户交互的脚本环境下执行 `cargo test`。解决这个问题的最佳方式，是实现一个合适的方法来关闭我们的操作系统——不幸的是，这个方式实现起来相对有些复杂，因为这要求我们实现对[APM](https://wiki.osdev.org/APM)或[ACPI](https://wiki.osdev.org/ACPI)电源管理标准的支持。

幸运的是，还有一个绕开这些问题的办法：QEMU支持一种名为 `isa-debug-exit` 的特殊设备，它提供了一种从客户系统（guest system）里退出QEMU的简单方式。为了使用这个设备，我们需要向QEMU传递一个 `-device` 参数。当然，我们也可以通过将 `package.metadata.bootimage.test-args` 配置关键字添加到我们的 `Cargo.toml` 来达到目的：
```toml
# in Cargo.toml

[package.metadata.bootimage]
test-args = ["-device", "isa-debug-exit,iobase=0xf4,iosize=0x04"]
```

`bootimage runner` 会在QEMU的默认测试命令后添加 `test-args` 参数。（对于 `cargo run` 命令，这个参数会被忽略。）

在传递设备名 (`isa-debug-exit`)的同时，我们还传递了两个参数，`iobase` 和 `iosize` 。这两个参数指定了一个_I/O 端口_，我们的内核将通过它来访问设备。

### I/O 端口

在x86平台上，CPU和外围硬件通信通常有两种方式，**内存映射I/O**和**端口映射I/O**。之前，我们已经使用内存映射的方式，通过内存地址 `0xb8000` 访问了[VGA文本缓冲区]。该地址并没有映射到RAM，而是映射到了VGA设备的一部分内存上。

与内存映射不同，端口映射I/O使用独立的I/O总线来进行通信。每个外围设备都有一个或数个端口号。CPU采用了特殊的`in`和`out`指令来和端口通信，这些指令要求一个端口号和一个字节的数据作为参数（有些这种指令的变体也允许发送 `u16` 或是 `u32` 长度的数据）。

`isa-debug-exit` 设备使用的就是端口映射I/O。其中， `iobase` 参数指定了设备对应的端口地址（在x86中，`0xf4` 是一个[通常未被使用的端口](https://wiki.osdev.org/I/O_Ports#The_list)），而 `iosize` 则指定了端口的大小（`0x04` 代表4字节）。

###   使用退出(Exit)设备

`isa-debug-exit` 设备的功能非常简单。当一个 `value` 写入 `iobase` 指定的端口时，它会导致QEMU以**退出状态**（[exit status](https://en.wikipedia.org/wiki/Exit_status)）`(value << 1) | 1` 退出。也就是说，当我们向端口写入 `0` 时，QEMU将以退出状态 `(0 << 1) | 1 = 1` 退出，而当我们向端口写入`1`时，它将以退出状态 `(1 << 1) | 1 = 3` 退出。

这里我们使用 [`x86_64`](https://docs.rs/x86_64/0.14.2/x86_64/) crate提供的抽象，而不是手动调用 `in` 或 `out` 指令。为了添加对该crate的依赖，我们可以将其添加到我们的 `Cargo.toml`中的 `dependencies` 小节中去:

```toml
# in Cargo.toml

[dependencies]
x86_64 = "0.14.2"
```
现在我们可以使用crate中提供的 [`Port`](https://docs.rs/x86_64/0.14.2/x86_64/instructions/port/struct.Port.html) 类型来创建一个 `exit_qemu` 函数了:
```rust
// in src/main.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

pub fn exit_qemu(exit_code: QemuExitCode) {
    use x86_64::instructions::port::Port;

    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
}
```
该函数在 `0xf4` 处创建了一个新的端口，该端口同时也是 `isa-debug-exit` 设备的 `iobase` 。然后它会向端口写入传递的退出代码。这里我们使用 `u32` 来传递数据，因为我们之前已经将 `isa-debug-exit` 设备的 `iosize` 指定为4字节了。上述两个操作都是 `unsafe` 的，因为I/O端口的写入操作通常会导致一些不可预知的行为。

为了指定退出状态，我们创建了一个 `QemuExitCode` 枚举。思路大体上是，如果所有的测试均成功，就以成功退出码退出；否则就以失败退出码退出。这个枚举类型被标记为 `#[repr(u32)]`，代表每个变量都是一个 `u32` 的整数类型。我们使用退出代码 `0x10` 代表成功，`0x11` 代表失败。 实际的退出代码并不重要，只要它们不与QEMU的默认退出代码冲突即可。 例如，使用退出代码0表示成功可能并不是一个好主意，因为它在转换后就变成了 `(0 << 1) | 1 = 1` ，而 `1` 是QEMU运行失败时的默认退出代码。 这样，我们就无法将QEMU错误与成功的测试运行区分开来了。

QEMU 定义了 **特定的退出码机制**：
- 0 → 默认不退出（有时会被解释为成功，也可能依赖于实现）
- 1 → 默认表示运行失败
现在我们来更新 `test_runner` 的代码，让程序在运行所有测试完毕后退出QEMU：
```rust
// in src/main.rs

fn test_runner(tests: &[&dyn Fn()]) {
    println!("Running {} tests", tests.len());
    for test in tests {
        test();
    }
    /// new
    exit_qemu(QemuExitCode::Success);
}
```
这里的问题在于，`cargo test` 会将所有非 `0` 的错误码都视为测试失败。

-  QEMU 通过 `isa-debug-exit` 写入退出码，然后 **bootimage runner** 捕获到这个状态码，并把它作为进程退出状态返回给 Cargo。
        
- Cargo 看到 `exit status != 0`，就认为测试失败。
###   成功退出(Exit)代码

为了解决这个问题， `bootimage` 提供了一个 `test-success-exit-code` 配置项，可以将指定的退出代码映射到退出代码 `0`:
```toml
# in Cargo.toml

[package.metadata.bootimage]
test-args = […]
test-success-exit-code = 33         # (0x10 << 1) | 1
```

我们的 test runner 现在会在正确报告测试结果后自动关闭QEMU。我们可以看到QEMU的窗口只会显示很短的时间——我们很难看清测试的结果。如果测试结果会打印在控制台上而不是QEMU里，让我们能在QEMU退出后仍然能看到测试结果就好了。
