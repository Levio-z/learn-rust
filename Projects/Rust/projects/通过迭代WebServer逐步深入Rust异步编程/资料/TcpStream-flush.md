“实际上不存在强制刷新网络套接字的方法，对 `TcpStream` 而言 `flush` 只是一个空操作，调用它是为了保持对 `io::Write` 的一致性。”
`TcpStream::flush()` 实际上什么都不做
Rust 的标准库中，`TcpStream` 实现了 `std::io::Write` trait：
```rust
impl Write for TcpStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { ... }
    fn flush(&mut self) -> io::Result<()> {
        // 这是空实现
        Ok(())
    }
}

```
`flush()` 被实现为 **no-op**（空操作）。
这是因为 **TCP 层面根本没有“刷新”这一概念**，不像 `BufWriter` 或 `stdout` 有缓存可以被主动清空。
### 为啥还要实现 `flush()`？
- `Write` trait 要求实现 `flush()` 方法。
- 虽然 `TcpStream` 不需要 flush，但为了实现 `Write` 接口，就必须提供这个函数。
- 所以给了一个“无害的空实现”。这让你可以把 `TcpStream`、`BufWriter<TcpStream>`、`File` 等统一看作 `Write` 类型处理，不需要特判。
### ❓那 TCP 何时发送数据？（flush 是空，数据怎么发的？）
TCP 的数据发送是由 **操作系统内核负责缓冲与传输** 的：
- 当你 `write()` 数据后，数据进入**socket 发送缓冲区**；
- 内核会**自动根据网络状态**、窗口大小、Nagle 算法等规则异步发送出去；
- **你无法直接“flush”网络**，除非你关闭连接（`shutdown`）或调整 socket 参数（例如关闭 Nagle）。

### 虽然不能直接 `flush()`，但你可以控制发送行为：
1. **关闭 Nagle 算法**（禁用合包，立刻发送）
	stream.set_nodelay(true)?;
	默认 TCP 会合并小包（Nagle 算法），禁用后写入就立即发送小包。
2. **写完后 shutdown 写端**
```rust
stream.shutdown(std::net::Shutdown::Write)?;
```
3. **使用 BufWriter 包裹 TcpStream，再手动 flush()**
```rust
let stream = TcpStream::connect(addr)?;
let mut writer = BufWriter::new(stream);
writer.write_all(b"Hello")?;
writer.flush()?; // 实际触发 write() 到 TcpStream

```
- `.write_all()` 最终会触发 `TcpStream::write()`；
- 因为频繁 `write()` 会触发系统调用，影响性能。`BufWriter` 的设计目标是：
- **减少系统调用次数**；
- 将多个小块数据一次写入底层流；
- 但你必须 `flush()`，否则数据还在用户层缓冲中，“还没送出去”。

### `write()` 做了什么？

调用 `TcpStream::write()` 实际上会：
1. 将数据从用户空间复制到 **内核态的 socket 发送缓冲区**（send buffer）；
2. 如果发送缓冲区未满，内核会立刻通过 TCP 协议把数据**异步地**发送出去；
3. 但是：这个“发送”可能是部分的，也可能是被缓冲。
🚫 注意：`write()` **不会告诉对方“我写完了”**，只是说“我还有数据”。

❗ 本质区别：`shutdown(Write)` 明确表示“我写完了”
 `shutdown(Write)` 的作用：
1. 除了发送缓冲区中的数据也会被发送外，
2. 还会附带一个 TCP **FIN 包** 发给对方，明确声明：
    > 🚪 **“我这边不会再写了，连接的写端关闭。”**
3. 对方在读的时候就能收到 EOF，或 read 返回 0，知道你已经写完。