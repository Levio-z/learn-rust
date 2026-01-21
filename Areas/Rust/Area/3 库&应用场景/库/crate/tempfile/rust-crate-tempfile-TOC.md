https://docs.rs/tempfile/3.22.0/tempfile/


```rust
use std::fs::File;
use std::io::{Write, Read, Seek, SeekFrom};

fn main() {
    // Write
    let mut tmpfile: File = tempfile::tempfile().unwrap();
    write!(tmpfile, "Hello World!").unwrap();

    // Seek to start
    tmpfile.seek(SeekFrom::Start(0)).unwrap();

    // Read
    let mut buf = String::new();
    tmpfile.read_to_string(&mut buf).unwrap();
    assert_eq!("Hello World!", buf);
}
```
### 场景
[创建一个临时文件，多次操作](场景/创建一个临时文件，多次操作.md)
