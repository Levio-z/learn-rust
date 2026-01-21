
### ptr::metadate
```rust
    pub const fn len(&self) -> usize {

        // ptr::metadata的作用就是从宽指针中提取元数据，对于切片来说，元数据就是长度（usize）。

        ptr::metadata(self)

    }
```