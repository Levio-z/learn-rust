
Copies the entire contents of a reader into a writer.


- **定义**：这是一个标准库的工具函数：
    
```rust
    pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64>
where
    R: Read,
    W: Write,

```
    
它会不断调用 `reader.read()` → 再调用 `writer.write_all()`，直到读完（遇到 EOF）。
    
- **返回值**：返回成功复制的总字节数（`u64`）。
    
- **这里的含义**：把 `buf` 中的数据（但最多 `size` 个字节，因为用了 `take`）复制到 `stdout`。

### 原理
```rust
let mut buffer = [0; 8*1024];
loop {
    let n = reader.read(&mut buffer)?;
    if n == 0 { break; }
    writer.write_all(&buffer[..n])?;
}
```
**注意**：
- `write_all` 会调用底层 writer 的 `write`
- 对 `StdoutLock` 来说，每次 `write` 已经是安全的写入
- **系统调用会立即写入**，不依赖缓冲器

- `Stdout` / `StdoutLock` 本身有内部缓冲（OS 层面缓冲），写入通常会立刻 flush
- Rust 的 `Stdout` 有 **行缓冲**（line-buffered）机制：
    - 换行符 `\n` 会触发 flush
    - 如果没有换行符，缓冲可能还在内核缓冲区中
- 对于 `StdoutLock`，手动 flush 只有在：
    1. 使用 `BufWriter` 缓冲写入
    2. 需要保证 **输出立即可见**  
        时才必要
