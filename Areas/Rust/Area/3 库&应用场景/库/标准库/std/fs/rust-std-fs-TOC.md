### 常见操作
#### 创建文件
##### 覆盖创建
- 文件不存在就创建  有就清空
```
std::fs::File
let mut f = File::create("foo.txt")?;
```
##### 追加创建
- 没有就创建（可选） 有就追加
```
std::fs::OpenOptions
let mut f = OpenOptions::new()
            .append(true)
            .create(true)
            .open("foo.txt")?;
```

- 一般用创建或清空文件的时候都传入write，这样不会出错
#### 读取文件
##### 打开+读取
打开文件-1:只读模式
```
std::fs::{File};
std::io::{Read};
let mut f = File::open("foo.txt")?;
let mut data = Vec::new();
f.read_to_end(&mut data)?;
let content = String::from_utf8_lossy(&data);
println!("read_to_end {:?}", content);
```
- 打开文件，读取为字节，找不到文件会报错
- `由于 Deref 的存在，Rust 允许把 &Vec<T> 自动解引用成 &[T]。`
##### 固定长度读取
```rust
let mut f = File::open("foo.txt")?;
let mut buf = [0u8; 4];
// 固定长度读取，必须填满，如果文件长度不足 4 字节 → 返回
// Err(ErrorKind::UnexpectedEof)
f.read_exact(&mut buf)?;
println!("{:?}", buf); // 输出 [0, 0, 0, 0]
```
##### 重置游标
```
std::io::{Seek, SeekFrom};
f.seek(SeekFrom::Start(0))?;
```
##### fs::read 便捷函数
>使用 File::open 和 read_to_end 且导入次数较少且没有中间变量的便捷函数
```rust
let mut f = fs::read("foo.txt")?;
let content = String::from_utf8_lossy(&f);
println!("read {:?}", content);
```
##### 读取为字符串
```rust
let content = fs::read_to_string("foo.txt")?;
println!("read_to_string :{} {}", content, content.len());
```
##### 读取行
```rust
// [读取行]
 let mut reader = BufReader::new(f);
let mut line = String::new();
// [读取多行]
for line in reader.lines().enumerate() {
 println!("line:{:?}---content:{:?}", line.0, line.1?);
}
```
#### 写入文件
```rust
// 返回写入的字数数，不保证一次都写入,需要检查数据确保全部写入：手动构造循环
let _ = f.write(&[56u8; 4])?;
// 内部构建循环，保证全部写入
f.write_all(&[56u8; 4])?;
f.write_all(b"hello world\n")?;
// 写入&str
f.write_all("中文".as_bytes())?;
// 写入宏
writeln!(&mut f, "hello world\n")？;
```
#### 权限的例子
- 一般情况下创建的文件没有读权限，需要的话需要使用fs下的OpenOptions设置

```rust
  let mut temp_dir = env::temp_dir();

        let temp_file = temp_dir.join("temp_file");

        let mut file: File = OpenOptions::new()

        .read(true)   // 开启读权限

        .write(true)  // 开启写权限

        .create(true) // 如果不存在就创建

        .truncate(true) // 如果存在就清空

        .open(&temp_file)?; // 打开文件

        // [writeln! 直接写入字符串]

        // 引入use std::io::Write;

        // 如果使用 直接 File（非 BufWriter），写操作已经在内核缓冲区，flush

        // 通常不是必须。

        writeln!(&mut file, "hello world\n")?;

  

        // 使用指针调整到开头才能读

        use std::io::{Seek, SeekFrom};

        file.seek(SeekFrom::Start(0))?;

  

        // [读取文件-4：fs::read_to_string]

        // 尝试读取

        let mut buf = String::new();

        let res = file.read_to_string(&mut buf);

        println!("read result: {:?}", buf); // Err(Bad file descriptor)

  

        fs::remove_file(&temp_file);
```
#### 读取目录
```rust
let mut dir = fs::read_dir(path).await.context("open directory failed")?;
```
#### 遍历目录的内容
```rust
        while let Some(entry) = dir.next_entry().await.context("read directory failed")? {

            let name = entry.file_name();

            let path = entry.path();

            let mode = Mode::from_path(&path).await?;

            vec.push((name, path, mode));

        }
```