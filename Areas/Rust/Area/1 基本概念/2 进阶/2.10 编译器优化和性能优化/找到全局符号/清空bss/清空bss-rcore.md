在内核初始化中，需要先完成对 `.bss` 段的清零。这是内核很重要的一部分初始化工作，在使用任何被分配到 `.bss` 段的全局变量之前我们需要确保 `.bss` 段已被清零。我们就在 `rust_main` 的开头完成这一工作，由于控制权已经被转交给 Rust ，我们终于不用手写汇编代码而是可以用 Rust 来实现这一功能了：
```rust
// os/src/main.rs
#[no_mangle]
pub fn rust_main() -> ! {
    clear_bss();
    loop {}
}

fn clear_bss() {
    extern "C" {
        fn sbss();
        fn ebss();
    }
    (sbss as usize..ebss as usize).for_each(|a| {
        unsafe { (a as *mut u8).write_volatile(0) }
    });
}
```
在函数 `clear_bss` 中，我们会尝试从其他地方找到全局符号 `sbss` 和 `ebss` ，它们由链接脚本 `linker.ld` 给出，并分别指出需要被清零的 `.bss` 段的起始和终止地址。接下来我们只需遍历该地址区间并逐字节进行清零即可。

extern “C” 可以引用一个外部的 C 函数接口（这意味着调用它的时候要遵从目标平台的 C 语言调用规范）。但我们这里只是**引用位置标志并将其转成 usize 获取它的地址**。由此可以知道 `.bss` 段两端的地址。