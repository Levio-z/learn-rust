

### 介绍
`Write` 是一个面向字节的输出（sink）接口。

实现了 `Write` trait 的对象有时被称为 **writer（写入器）**。

Writer 由两个必需的方法定义：`write` 和 `flush`：

- `write` 方法会尝试向对象写入一些数据，并返回成功写入的字节数。
    
- `flush` 方法通常用于适配器或带显式缓冲的 writer，确保所有缓冲区中的数据被推送到真正的“底层输出”。
    

Writer 的设计目标是可以彼此组合。Rust 标准库中很多 [`std::io`](self) 的实现都接收或返回实现了 `Write` trait 的类型。
### 如何实现
在 Rust 的 `std::io::Write` trait 中，你看到的方法很多，其实分成两类：
1. **必须实现的方法（required methods）**
    - `fn write(&mut self, buf: &[u8]) -> Result<usize>`
    - `fn flush(&mut self) -> Result<()>`
    这两个方法是 **核心接口**，它们没有默认实现，必须由具体类型提供实现。
2. **默认实现的方法（provided methods）**
    - `write_all`、`write_fmt`、`write_vectored`、`write_all_vectored`、`by_ref` 等
    - 这些方法是基于 **核心方法** `write` 和 `flush` 实现的 **默认逻辑**。
    - 比如 `write_all` 就是循环调用 `write` 直到把所有数据写完。

因此，当你为自定义类型（如 `HashWriter`）实现 `Write` 时，只要定义 `write` 和 `flush`，其它方法自动继承默认实现即可。
[1.2 Rust中的IO装饰器，流式写入同步计算设计](../../../../../../../../../../Projects/设计模式/23经典设计模式/结构型/装饰器模式/1.2%20Rust中的IO装饰器，流式写入同步计算设计.md)