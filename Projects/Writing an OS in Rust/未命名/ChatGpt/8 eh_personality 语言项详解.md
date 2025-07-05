`eh_personality` 语言项标记的函数，将被用于实现**栈展开**（[stack unwinding](https://www.bogotobogo.com/cplusplus/stackunwinding.php)）。在使用标准库的情况下，当 panic 发生时，Rust 将使用栈展开，来运行在栈上所有活跃的变量的**析构函数**（destructor）——这确保了所有使用的内存都被释放，允许调用程序的**父进程**（parent thread）捕获 panic，处理并继续运行。但是，栈展开是一个复杂的过程，如 Linux 的 [libunwind](https://www.nongnu.org/libunwind/) 或 Windows 的**结构化异常处理**（[structured exception handling, SEH](https://docs.microsoft.com/en-us/windows/win32/debug/structured-exception-handling)），通常需要依赖于操作系统的库；所以我们不在自己编写的操作系统中使用它。**

### 1. 定义与作用

`eh_personality` 是 Rust 中一个**语言项（lang item）**，用于标记实现异常处理个性化函数（personality function）的函数。

- **栈展开（stack unwinding）**过程的关键部分就是调用此函数。
    
- 它负责协助运行时确定异常传播过程中的控制流，保证 **panic 发生时栈上所有活跃变量的析构函数都能被调用**，实现正确资源回收。
    

---

### 2. 工作原理

当 Rust 程序发生 panic，且启用了栈展开（默认模式），运行时会：

1. 调用 `eh_personality` 标记的函数，查询如何处理异常。
    
2. 逐层展开调用栈，运行每个栈帧中局部变量的 `Drop`，以释放资源。
    
3. 最终将异常传递给调用者（如父线程）或者终止程序。
    

底层这个过程依赖：

- Linux 平台一般依赖 `libunwind` 库，基于 DWARF 异常处理信息实现展开。
    
- Windows 平台依赖结构化异常处理（SEH）。
    
- 运行时环境提供异常处理机制。
    

---

### 3. 在操作系统或裸机环境的限制

对于自己编写的操作系统或裸机程序（如基于 Rust 的内核、bootloader、嵌入式系统）：

- 缺少操作系统的异常处理库支持。
    
- 复杂的栈展开机制难以移植和实现。
    
- 资源受限，栈展开增加开销。
    

因此，通常选择**关闭栈展开**，用 `panic = "abort"` 策略替代。

---

### 4. Rust 生态中的实践

- 标准库依赖 `eh_personality` 来支持 unwind。
    
- `#![no_std]` 或裸机环境通常不实现 `eh_personality`，也不启用 unwind。
    
- 相关语言项报错时，常见解决方案是设置：
    

toml

复制编辑

`[profile.dev] panic = "abort"`

并提供自定义的 `panic_handler`。