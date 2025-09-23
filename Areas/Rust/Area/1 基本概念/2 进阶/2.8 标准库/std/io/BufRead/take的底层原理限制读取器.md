https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=a4da89eb6ba9875c16e96d15c40dc8d1
```rust
use std::io::{self, Read};

/// LimitReader 定义（复用你的实现）
struct LimitReader<R> {
    reader: R,
    limit: usize,
}

impl<R: Read> Read for LimitReader<R> {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        if buf.len() > self.limit {
            buf = &mut buf[..self.limit];
        }
        let n = self.reader.read(buf)?;
        if n > self.limit {
            return Err(io::Error::new(io::ErrorKind::Other, "read limit exceeded"));
        }
        self.limit -= n;
        Ok(n)
    }
}

fn main() -> io::Result<()> {
    let data = b"Hello, Rust! This is a test."; // 测试数据
    let cursor = std::io::Cursor::new(data);  // 内存缓冲实现 Read trait

    let mut reader = LimitReader {
        reader: cursor,
        limit: 10, // 设置最多读取 10 个字节
    };

    let mut buf = Vec::new();
    let n = reader.read_to_end(&mut buf)?; // 尝试读取剩余数据
    println!("实际读取了 {} 个字节: {:?}", n, String::from_utf8_lossy(&buf));

    // 再次读取，应该返回 0（或 EOF）
    let mut buf2 = [0u8; 5];
    let n2 = reader.read(&mut buf2)?;
    println!("再次读取 {} 个字节: {:?}", n2, &buf2[..n2]);

    Ok(())
}


```
### 关键点解释

1. **`buf: &mut [u8]` 截断**
    - `read` 要求填充整个缓冲区，但我们要做限制，所以用切片操作 `&mut buf[..self.limit]` 来缩小读取范围。
        
2. **错误处理**
    - 如果读取字节数超过 `self.limit`，就返回 `io::Error`。
    - 用 `io::Error::new(io::ErrorKind::Other, "read limit exceeded")` 创建一个新的错误。
3. **扣减剩余 limit*
    - 每次读完后，更新 `self.limit`，保证总读取量不超过限制。

装饰器模式 = 保持接口 + 增加行为。
### 读取器
最外面的装饰者需要可变，但是被包装的不需要可变