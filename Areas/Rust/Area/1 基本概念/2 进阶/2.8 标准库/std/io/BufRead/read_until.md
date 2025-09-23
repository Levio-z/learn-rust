
`read_until` 是 Rust 标准库 `std::io::BufRead` trait 提供的方法，用于按 **指定分隔符** 读取输入流，适合逐行或按特定分隔符处理数据。

---
### 1. 方法定义

`fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> Result<usize>`

- **参数**
    1. `byte: u8` – 分隔符（delimiter），读取直到遇到这个字节
    2. `buf: &mut Vec<u8>` – 输出缓冲区，读取的数据会 **追加** 到 `buf`
- **返回值**
    - `Ok(n)`：实际读取的字节数（包括分隔符，如果遇到了）
    - `Err(e)`：I/O 错误
---

### 2. 工作原理

1. 从流中读取数据，直到遇到指定分隔符或流结束（EOF）
    
2. 将读取到的字节追加到传入的 `buf`
    
3. **返回读取的总字节数**
    

关键特性：

- **累加**：每次调用 `read_until` 会把新读取的数据追加到 `buf`，不会覆盖原内容
    
- **分隔符包含在结果中**：如果找到分隔符，它会被包含在 `buf` 的末尾
    
- **EOF 处理**：
    
    - 如果 EOF 且 `buf` 仍为空，返回 `Ok(0)`
        
    - 否则返回 EOF 前读取的字节数
        

---

### 3. 示例

```rust
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let data = b"hello\nworld\nrust";
    let mut reader = &data[..]; // &[u8] 实现 BufRead
    let mut buf = Vec::new();

    while reader.read_until(b'\n', &mut buf)? != 0 {
        println!("Line: {:?}", String::from_utf8_lossy(&buf));
        buf.clear(); // 清空缓冲区以读取下一行
    }

    Ok(())
}


```

**输出：**

`Line: "hello\n" Line: "world\n" Line: "rust"`

注意：

- `b'\n'` 是分隔符
    
- 最后一行没有 `\n`，仍然被读出