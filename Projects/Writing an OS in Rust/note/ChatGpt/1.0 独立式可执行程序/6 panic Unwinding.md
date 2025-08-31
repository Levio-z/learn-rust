## 1. 什么是“展开堆栈”（Unwinding）？

- 当 Rust 遇到 `panic!` 时，它会从触发 panic 的位置开始“展开调用栈”。
- 具体来说，程序会沿着调用链逐层返回，执行每个作用域中的 `Drop` 清理代码（类似析构函数），释放资源。
- 这个过程就是“堆栈展开”（stack unwinding），相当于异常处理机制中“清理现场”。
- 这个展开过程可以让程序有机会优雅地释放资源，防止内存泄漏。

---

## 2. panic 触发是否一定终止程序？

- **默认情况下，panic 会导致线程终止**，但不一定让整个程序崩溃。
- Rust 支持**捕获 panic**，通过 `std::panic::catch_unwind`，你可以在程序中捕获 panic，防止程序崩溃。
    

```rust
use std::panic;

fn main() {
    let result = panic::catch_unwind(|| {
        panic!("something went wrong!");
    });

    assert!(result.is_err());
    println!("panic caught, program continues");
}

    
```

---

## 3. 什么时候 panic 会终止整个程序？

- 当 panic **未被捕获**时，线程会终止。
    
- 对于主线程，未捕获 panic 会导致整个程序终止（退出）。
    
- 对于其他线程，panic 会导致该线程终止，但主线程继续执行。
    
- 如果 panic 发生在 `Drop` 代码中并且正在展开堆栈，会触发“二次 panic”，此时 Rust 会 **调用 abort，直接终止整个程序**。
    

---

## 4. Rust 支持两种 panic 策略

|策略|说明|影响|
|---|---|---|
|unwind|默认策略，展开堆栈释放资源|支持捕获 panic，线程终止但程序可继续运行|
|abort|直接终止程序，不展开堆栈|无法捕获 panic，panic 立即导致进程退出|
### 5、不支持展开的少数情况
    
- **在 `Drop` 实现中发生 panic 时**，如果 `Drop` 已经处于展开状态时再次 panic，就会触发 **二次 panic（double panic）**，Rust 设计为**立即 abort** 程序，避免不可恢复的状态。
- **函数的 ABI 不支持展开时**（如 C 函数 `extern "C"`）：
	- Rust 的 FFI（与 C 接口）中，默认不支持异常展开进入或离开 `extern "C"` 函数。这是为了 **兼容 C 语言不支持栈展开的模型**。
```

#[no_mangle]
pub extern "C" fn my_ffi_func() {
    panic!("cannot unwind through C ABI!");
    // ❗ 若开启 unwind，会触发 undefined behavior 或 abort
}
```
### 6、不建议展开硬要展开
panic 处理程序也可以安全地展开，但这只会导致 panic 处理程序再次被调用。
- 当检测到“二次 panic”时，panic handler 递归调用自己；超出允许递归深度（2）后，调用 `abort` 终止程序，确保安全性和稳定性。