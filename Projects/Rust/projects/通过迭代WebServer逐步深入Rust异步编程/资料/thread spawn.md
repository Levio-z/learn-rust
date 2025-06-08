```rust
std::thread::spawn(|| {
    if let Err(e) = handle_connection(connection) {
        println!("failed to handle connection: {e}")
    }
});

```
`std::thread::spawn` 是 Rust 标准库中用于**创建新线程**的方法。
- 这个函数接受一个闭包 `|| { ... }` 作为参数，这个闭包的内容将在**新线程中执行**。
- 在你的例子中，它的作用是：**为每一个客户端连接创建一个线程来处理请求**，从而实现并发处理多个连接。

`std::thread::spawn` 底层基于操作系统提供的线程接口（如：
- Linux 的 `pthread_create`
- Windows 的 `_beginthreadex`
- macOS 的 `pthread_create`
它会创建一个独立的系统线程，堆栈空间默认大小（如 2MB，可配置），并把闭包作为线程入口函数执行。
### 源码
```rust
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static

```
- ❗️ 看到没？闭包 `F` 必须是 `'static`
- 原因是线程的运行时间不确定，可能会在调用线程返回后仍继续运行，因此闭包中捕获的数据必须是 `'static` 生命周期

