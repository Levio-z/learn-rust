在使用 Rust 编写应用程序的时候，我们常常在遇到了一些无法恢复的致命错误（panic），导致程序无法继续向下运行。**这时手动或自动调用 `panic!` 宏来打印出错的位置，让软件能够意识到它的存在，并进行一些后续处理**。 `panic!` 宏最典型的应用场景包括断言宏 `assert!` 失败或者对 `Option::None/Result::Err` 进行 `unwrap` 操作。所以Rust编译器在编译程序时，从安全性考虑，需要有 `panic!` 宏的具体实现

在标准库 std 中提供了关于 `panic!` 宏的具体实现，其大致功能是**打印出错位置和原因并杀死当前应用**。但本章要实现的操作系统不能使用还需依赖操作系统的标准库std，**而更底层的核心库 core 中只有一个 `panic!` 宏的空壳，并没有提供 `panic!` 宏的精简实现**。因此我们需要自己先实现一个简陋的 panic 处理函数，这样才能让“三叶虫”操作系统 – LibOS的编译通过。


### `#[panic_handler]`

`#[panic_handler]` 是一种编译指导属性，用于**标记核心库core中的 `panic!` 宏要对接的函数**（该函数实现对致命错误的具体处理）。这样Rust编译器就可以把核心库core中的 `panic!` 宏定义与 `#[panic_handler]` 指向的panic函数实现合并在一起，使得no_std程序具有类似std库的应对致命错误的功能。
- 函数签名：该编译指导属性所标记的函数需要具有 `fn(&PanicInfo) -> !` 函数签名，函数可通过 `PanicInfo` 数据结构获取致命错误的相关信息。

我们创建一个新的子模块 `lang_items.rs` 实现panic函数，并通过 `#[panic_handler]` 属性通知编译器用panic函数来对接 `panic!` 宏。为了将该子模块添加到项目中，我们还需要在 `main.rs` 的 `#![no_std]` 的下方加上 `mod lang_items;` ，相关知识可参考 [Rust 模块编程](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter1/2remove-std.html#rust-modular-programming) ：

```rust
// os/src/lang_items.rs
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
```
在把 `panic_handler` 配置在单独的文件 `os/src/lang_items.rs` 后，需要在os/src/main.rs文件中添加以下内容才能正常编译整个软件：
```rust
// os/src/main.rs
#![no_std]
mod lang_items;
// ... other code
```

注意，panic 处理函数的函数签名需要一个 `PanicInfo` 的不可变借用作为输入参数，它在核心库中得以保留，这也是我们第一次与核心库打交道。之后我们会从 `PanicInfo` 解析出错位置并打印出来，然后杀死应用程序。但目前我们什么都不做只是在原地 `loop` 。
### 拓展
-  [panic标准库是怎么做的](https://www.notion.so/panic-1d755e4de82481ab8d1ecb0e4c2f80de?pvs=21)