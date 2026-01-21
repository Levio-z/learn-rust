```rust
use std::io::prelude::*;
use flate2::Compression;
use flate2::write::ZlibEncoder;

// Vec<u8> implements Write, assigning the compressed bytes of sample string

let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
e.write_all(b"Hello World")?;
let compressed = e.finish()?;
```
参数传入的是一个writer
[flush](flush.md)