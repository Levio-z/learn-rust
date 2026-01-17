---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 一个基本示例
- 本质:
    - 相当于其他语言的interface
    - 与泛型密不可分
- 组成:
    - Required methods(必须实现)
    - Provided methods(可选实现)
- 示例分析:
    - Read trait必须实现read方法
    - Write trait必须实现write和flush方法
    - 方法签名包含&mut self表示可变借用
```rust
/// 读取数据的 trait（对应标准库 std::io::Read）
pub trait Read {
    // 必须实现的核心方法：从自身读取数据到 buf
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    // 可选方法（默认实现）：从自身读取数据到多个缓冲区（分散读取）
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> Result<usize> {
        let mut total_read = 0;
        for buf in bufs {
            let n = self.read(buf)?;
            total_read += n;
            if n == 0 { break; } // 没有更多数据则终止
        }
        Ok(total_read)
    }

    // 标准库中还有 read_to_end、read_exact 等方法，这里省略
}


/// 写入数据的 trait（对应标准库 std::io::Write）
pub trait Write {
    // 必须实现的核心方法1：将 buf 数据写入自身
    fn write(&mut self, buf: &[u8]) -> Result<usize>;
    // 必须实现的核心方法2：刷新缓冲区（确保数据真正写入目标）
    fn flush(&mut self) -> Result<()>;

    // 可选方法（默认实现）：将多个缓冲区数据写入自身（集中写入）
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> Result<usize> {
        let mut total_written = 0;
        for buf in bufs {
            let n = self.write(buf)?;
            total_written += n;
            if n == 0 { break; } // 无法写入更多则终止
        }
        Ok(total_written)
    }

    // 标准库中还有 write_all、write_fmt 等方法，这里省略
}
```
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-trait bound-基本概念](trait_bound/Rust-trait%20bound-基本概念.md)
	- [Rust-trait-rust提供的](Rust-trait-rust提供的.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
