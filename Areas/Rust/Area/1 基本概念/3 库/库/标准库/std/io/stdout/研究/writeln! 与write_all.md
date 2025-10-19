
```rust
let stdout = std::io::stdout();
// 自带行缓冲
let mut stdout = stdout.lock();

write!(
&mut stdout,
"{mode:0>6} {} {}  {name}",
 kind,
 hex::encode(hashbuf),
)?;

stdout.write_all(b"\n")?;
```
此例可以合并为writeln!,只是展示使用示例

输出使用std::io::stdout多次写入终端，避免多次系统调用和交错输出
- writeln!有格式化开销
- 单字节/固定字节序列的情况使用stdout.write_all(b"\n")?;