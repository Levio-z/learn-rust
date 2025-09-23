### 基本定义

### 定义与作用
- **定义**  
    `write(&mut self, buf: &[u8]) -> Result<usize>
    - **把 `buf` 中的字节写入到当前 write**r（实现 `Write` trait 的对象）。
    - 返回值是一个 `Result<usize>`，其中：
        - `Ok(n)`：表示成功写入了 `n` 个字节（`0 <= n <= buf.len()`）。
        - `Err(e)`：表示写入过程中出现了 I/O 错误。
		    - 如果无法将整个缓冲区写入此写入器，则**不会将**其视为错误。
		    - 每次调用`写`操作都可能生成一个 I/O 错误，指示操作无法完成。如果返回错误，则缓冲区中没有字节写入此写入器。
- **作用**  
    提供一个抽象接口，让各种写目标（文件、网络 socket、内存 buffer、标准输出等）都能以统一方式写入字节。
### 细节解释
1. **不保证写入全部数据**
    - 一次调用 `write` 可能只写入部分字节（例如 socket 的发送缓冲区满了）。
    - 这是 **“部分写” (partial write)** 语义。
    - 如果要保证写入全部内容，应该使用 `write_all`。
2. **`Ok(0)` 的含义**
    - 如果返回 `Ok(0)`，通常表示目标不能再接收字节（比如文件到达 EOF，或者 socket 已经关闭）。
3. **错误处理**
    - 如果发生错误（`Err`），则没有任何字节被写入。
    - `ErrorKind::Interrupted` 特殊：表示操作被中断（如信号打断），此时应当 **重试**。
4. **非阻塞语义**
    - 对于网络和某些设备，调用 `write` 不一定会阻塞等待。
    - 例如非阻塞 socket 可能立即返回 `Err(WouldBlock)`。
### `Write` trait 的默认方法

| 方法                                        | 描述                                  |
| ----------------------------------------- | ----------------------------------- |
| `write_all(&mut self, buf: &[u8])`        | 尝试写入整个缓冲区，内部循环直到全部写完或出错             |
| `write_fmt(&mut self, fmt: Arguments)`    | 支持格式化写入，相当于 `format!` + `write_all` |
| `by_ref()`                                | 返回对 `Write` 的可变引用，可链式调用             |
| `chain()`                                 | 将两个写入目标链接起来，写入时依次写入两个目标             |
| write!(writer, "blob {}\0", stat.len())?; | 接受格式化写入                             |

### 谁实现了 `Write`？

| 类型                          | 描述        |
| --------------------------- | --------- |
| `std::fs::File`             | 文件写入      |
| `std::net::TcpStream`       | 网络发送      |
| `std::io::Stdout`、`Stderr`  | 标准输出、标准错误 |
| `Vec<u8>`、`Cursor<Vec<u8>>` | 内存缓冲区     |
### 源码原理

Rust 标准库里 `Write::write` 是一个 **trait 方法**，不同的 writer（如 `File`、`TcpStream`、`Vec<u8>`）会有不同实现：
- 对 **文件 (`File`)**，它底层调用操作系统的 `write` 系统调用。
- 对 **内存 buffer (`Vec<u8>`)**，直接在内存中追加字节。
- 对 **网络 socket (`TcpStream`)**，最终调用 OS 的网络 I/O 接口。
底层的系统调用或驱动，决定了能写多少字节、是否会阻塞、是否报错。
### 使用场景

- **文件写入**：`File::create("foo.txt")?.write(b"hello")?;`
- **内存 buffer**：`let mut buf = Vec::new(); buf.write(b"abc")?;`
- **网络传输**：`TcpStream.write(b"GET / HTTP/1.1\r\n")?;`
```
writer.write_all(b"entire message")?;
```

### 扩展知识点
- **`write` vs `write_all`**
    - `write`: 可能部分写入。
    - `write_all`: 保证写完整（循环调用 `write`，直到完成或错误）。
- **`BufWriter`**
    - 提供缓冲写入，减少系统调用次数，提高性能。
- **`AsyncWrite` (Tokio)**
    - 异步 I/O 的 `write`，返回 `Poll<Result<usize>>`，适用于高并发网络编程。
### 总结
- `write` 是 Rust I/O 写入的基础接口，核心特点是 **可能部分写入**。
- 返回值 `Ok(n)` 表示成功写入 n 字节，`Ok(0)` 代表不可再写入。
- 错误时没有写入任何数据，`Interrupted` 错误应当重试。
- 想确保完整写入，必须使用 `write_all`。

