```rust
// os/src/main.rs
#[no_mangle]
pub fn rust_main() -> ! {
    loop {}
}
```
这里需要注意的是需要通过宏将 `rust_main` 标记为 `#[no_mangle]` 以避免编译器对它的名字进行混淆，不然在链接的时候， `entry.asm` 将找不到 `main.rs` 提供的外部符号 `rust_main` 从而导致链接失败。在 `rust_main` 函数的开场白中，我们将第一次在栈上分配栈帧并保存函数调用上下文，它也是内核运行全程中最底层的栈帧。