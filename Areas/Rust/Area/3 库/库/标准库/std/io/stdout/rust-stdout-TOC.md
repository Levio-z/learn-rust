```
let mut stdout = std::io::stdout().lock();
```
- **定义**：获取当前进程的标准输出句柄，并加锁返回一个 `StdoutLock`。
- **为什么要锁**：
    - `stdout()` 本身是一个共享的全局句柄。
    - 在多线程环境下直接操作 `stdout` 可能发生竞争。
    - `lock()` 提供一个临时独占的可写视图，使得写操作是**原子化**的，不会被其他线程的输出打断。
- **返回值**：`StdoutLock<'_>`，实现了 `Write` trait。

[read 和 stdout()的使用指南](研究/read%20和%20stdout()的使用指南.md)
[writeln! 与write_all](研究/writeln!%20与write_all.md)