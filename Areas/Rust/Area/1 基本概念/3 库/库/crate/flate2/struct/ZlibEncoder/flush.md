#### 1. 完成压缩流（flush 压缩器状态）

- `ZlibEncoder` 内部持有一个 **压缩状态机（zlib stream / `Deflate` 状态）**。
    
- 在正常写入时（`write_all` 或 `write`），压缩器将数据分块压缩，输出到下层 `writer`。
    
- `finish()` 会：
    
    1. 将压缩器内部缓存的剩余数据全部处理（flush 内部缓冲区）。
    2. 写入 **Zlib 流的结尾标记**（Zlib header + Deflate block + Adler32 校验和）。
>换句话说，`finish()` 会确保压缩流完整，生成可以被标准解压工具正确解析的 Zlib 数据。

#### 2. 写入尾部校验（Adler32）

- Zlib 数据格式要求：
    
    `[2-byte Zlib header] + [Deflate 压缩块] + [4-byte Adler32 校验和]`
    
- `finish()` 会：
    
    - 调用内部压缩器将剩余数据压缩成 Deflate 块
        
    - 计算整个输入流的 **Adler32** 校验
        
    - 写入压缩流末尾，保证解压完整性

#### 3. 返回原始写入器

- 调用 `finish()` 后，会返回内部原始写入器（`writer`）：
    
    `let inner_writer = writer.finish()?;`
    
- 此时压缩器对象被消费（drop），内部状态清理完毕。
    
- 原始写入器可以继续使用，比如写入其他数据、关闭文件等。

#### 4. 底层逻辑简化流程

1. flush 内部缓冲（可能有未压缩的数据）
    
2. 压缩剩余数据块
    
3. 写入 Zlib 结尾标记
    
4. 计算并写入 Adler32 校验
    
5. 消费 `ZlibEncoder`，返回原始写入器