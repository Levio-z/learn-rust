

```java
.. code-block::

   $ cargo build
      Compiling os v0.1.0 (/home/shinbokuow/workspace/v3/rCore-Tutorial-v3/os)
   error: requires `start` lang_item
```

- 新版报错
    
    ```java
    error: using `fn main` requires the standard library
      |
      = help: use `#![no_main]` to bypass the Rust generated entrypoint and declare a platform specific entrypoint yourself, usually with `#[no_mangle]`
    ```

- 在普通 Rust 程序里，编译器默认会生成一个 **入口函数 (entrypoint)**，即 `mainCRTStartup → main → fn main()` 的调用链。
- 这个入口逻辑依赖 **标准库 `std`**（因为它要设置运行时环境，例如栈溢出保护、线程、I/O）。
- 当你使用 `#![no_std]` 时，`std` 不可用，Rust 就无法自动生成 `main` 的启动逻辑。
- 所以编译器报错：**使用 `fn main` 需要标准库**。

编译器提醒我们缺少一个名为 `start` 的语义项。我们回忆一下，之前提到语言标准库和三方库作为应用程序的执行环境，需要负责在执行应用程序之前进行一些初始化工作，然后才跳转到应用程序的入口点（也就是跳转到我们编写的 `main` 函数）开始执行。事实上 `start` 语义项代表了标准库 std 在执行应用程序之前需要进行的一些初始化工作。由于我们禁用了标准库，编译器也就找不到这项功能的实现了。

最简单的解决方案就是压根不让编译器使用这项功能。我们在 `main.rs` 的开头加入设置 `#![no_main]` 告诉编译器我们没有一般意义上的 `main` 函数，并将原来的 `main` 函数删除。在失去了 `main` 函数的情况下，编译器也就不需要完成所谓的初始化工作了。


```rust
cargo build --target riscv64gc-unknown-none-elf
```
至此，我们成功移除了标准库的依赖，并完成了构建裸机平台上的“三叶虫”操作系统的第一步工作–通过编译器检查并生成执行码。

本小节我们固然脱离了标准库，通过了编译器的检验，但也是伤筋动骨，将原有的很多功能弱化甚至直接删除，看起来距离在 RV64GC 平台上打印 `Hello world!` 相去甚远了（我们甚至连 `println!` 和 `main` 函数都删除了）。不要着急，接下来我们会以自己的方式来重塑这些基本功能，并最终完成我们的目标。

