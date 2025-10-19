### 使用
```rust
 let content = fs::read(path).unwrap();
```
### 简介
`std::fs::read` 是 Rust 标准库中提供的一个 **便捷函数**，用于一次性将文件的全部内容读取到 `Vec<u8>` 中。它隐藏了文件打开、读取缓冲、循环调用 `read` 等细节，让调用者只需要提供一个路径即可获取文件的完整字节内容
```rust
#[stable(feature = "fs_read_write_bytes", since = "1.26.0")]
pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    fn inner(path: &Path) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?; // ① 打开文件
        let size = file.metadata()
            .map(|m| usize::try_from(m.len()).unwrap_or(usize::MAX))
            .ok(); // ② 读取文件大小（字节数）
        let mut bytes = Vec::try_with_capacity(size.unwrap_or(0))?; // ③ 预分配内存
        io::default_read_to_end(&mut file, &mut bytes, size)?; // ④ 读文件直到 EOF
        Ok(bytes) // ⑤ 返回 Vec<u8>
    }
    inner(path.as_ref())
}
```
### 逐行解读

#### ① `File::open(path)?`

- 调用 [`std::fs::File::open`] 打开文件，返回 `Result<File>`。
- 如果文件不存在或权限不足，直接返回 `Err(io::Error)`。
#### ② `file.metadata()...`
- 调用 `file.metadata()` 获取文件元数据（`Metadata`），其中包含文件大小（`len()` 返回字节数）。
- 将文件大小 (`u64`) 转换为 `usize`。
    - 如果文件过大（大于 `usize::MAX`），则用 `usize::MAX` 替代。
- 如果获取 `metadata` 失败（比如某些虚拟文件系统），`ok()` 会把错误丢弃，返回 `None`。
#### ③ `Vec::try_with_capacity(size.unwrap_or(0))?`

- 根据文件大小预分配 `Vec<u8>` 的容量，避免在读文件过程中频繁扩容。
- 如果无法分配内存（如文件过大，内存不足），返回 `Err`.

#### ④ `io::default_read_to_end(&mut file, &mut bytes, size)?`

- 循环调用 [`Read::read`]，直到 EOF。
- 内部会处理 `io::ErrorKind::Interrupted`，自动重试。
- 把数据依次写入 `bytes`。
- `size` 参数用于优化扩容策略：
    - 如果提前知道文件大小，可以减少 `Vec` 的扩容次数。
#### ⑤ `Ok(bytes)`
- 最终返回 `Vec<u8>`，包含完整文件内容。