-  `std::sync::Once`
- 以前，`Once` 是标准库中唯一用于“一次性执行”的同步原语。  在 `OnceLock<T>` 和 `LazyLock<T, F>` 被正式纳入标准库之前，其他库曾基于 `Once` 实现了这些更高级的同步结构。  尤其是 `OnceLock<T>`，它在功能上已完全取代了 `Once` 的主要用途。  在多数 `Once` 需要与数据绑定的场景中，**更推荐使用 `OnceLock<T>`**。
- 此类型只能使用 [`Once：：new（）`](https://doc.rust-lang.org/std/sync/struct.Once.html#method.new "associated function std::sync::Once::new") 构造。

## Examples  例子
```rust
#![allow(unused)]
fn main() {
    use std::sync::Once;

    static START: Once = Once::new();

    START.call_once(|| {
        // run initialization here
        println!("你好1");
    });
    START.call_once(|| {
        // run initialization here
        println!("你好2");
    });
}


```
结果：
```
你好1
```
### call_once
```rust
pub fn call_once<F>(&self, f: F) 
where
    F: FnOnce(),

```
执行一次初始化例程，并且只执行一次。如果这是第一次调用 call_once，则将执行给定`的`闭包，否则_不会_调用例程。
如果另一个初始化例程当前正在运行，则此方法将阻止调用线程。
当此函数返回时，可以保证某些初始化已运行并完成（可能不是指定的闭包）。还可以保证此时其他闭包执行完毕后，其他线程能可靠地“看到”闭包中的写入操作（例如初始化的数据）。这是**内存屏障**和**同步原语的 happens-before 保证**的体现。
如果给定的闭包递归地调用同一个 [`Once`](https://doc.rust-lang.org/std/sync/struct.Once.html "struct std::sync::Once") 上的 `call_once` 实例，则未指定确切的行为：允许的结果是 恐慌或死锁。
```rust
static mut VAL: usize = 0;
static INIT: Once = Once::new();

fn get_cached_val() -> usize {
    unsafe {
        INIT.call_once(|| {
            VAL = expensive_computation(); // ✅ 只会被执行一次
        });
        VAL // ✅ 所有线程都可以读取这个结果
    }
}

```
- `static mut` 不安全，必须通过同步机制保护（此处使用 `Once`）
- `call_once` 保证 `VAL` 只会被初始化一次，即使多个线程同时调用 `get_cached_val()`
- 返回值会被缓存住（“惰性初始化”）
### 内部状态机
```rust
INCOMPLETE
   │
call_once()
   ▼
RUNNING → 如果 panic → POISONED
   │
   ▼
正常完成 → COMPLETE

```
- `call_once()` 或 `call_once_force()` 被成功执行一次后，状态从 `INCOMPLETE → RUNNING → COMPLETE`
- 一旦状态为 `COMPLETE`，**无论是 `call_once()` 还是 `call_once_force()`，闭包都不再执行**