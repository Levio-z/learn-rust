### 1 参考
#### 1.1 挑战内容
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-01-gg4.md
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-02-ic4.md
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-03-jt4.md
#### 1.2 资料

[Git对象](https://git-scm.com/book/zh/v2/Git-%e5%86%85%e9%83%a8%e5%8e%9f%e7%90%86-Git-%e5%af%b9%e8%b1%a1)：你是否好奇git add. , git commit的底层原理，跟着这个可以很快理解内部原理和存储过程
[Git引用](https://git-scm.com/book/zh/v2/Git-%e5%86%85%e9%83%a8%e5%8e%9f%e7%90%86-Git-%e5%bc%95%e7%94%a8)
#### 1.3 知识点
- 文件相关：[rust-std-fs-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/fs/rust-std-fs-TOC.md)
- zlibDecoder：[ZlibEncoder-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/flate2/struct/ZlibEncoder/ZlibEncoder-TOC.md)
- 什么是c风格字符串：[Rust  CStr TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/ffi/Cstr/Rust%20%20CStr%20TOC.md)
- 分割字符串：[Rust `&str` 常用字符串操作方法概览](../../../../Areas/Rust/Area/3%20库/库/标准库/core/str/常用方法/Rust%20`&str`%20常用字符串操作方法概览.md)
- 装饰器（限制读取）：[take](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/BufRead/take.md)
- 解析为数值：[解析为数值](../../../../Areas/Rust/Area/3%20库/库/标准库/core/str/常用方法/解析为数值.md)
- print的底层：[rust-stdout-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/stdout/rust-stdout-TOC.md)
	- [copy reader to writer](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/methods/io%20copy/copy%20reader%20to%20writer.md)
- 错误处理： [rust-anyhow-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/anyhow/rust-anyhow-TOC.md)
- 临时文件：[rust-crate-tempfile-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/tempfile/rust-crate-tempfile-TOC.md)
- 命令行工具：[rust-clap-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/clap/rust-clap-TOC.md)
- 文件格式和基本流程梳理(建议单独开启一个网址来参考)：[Git Object对象](../../../../Areas/basic/Git/Pro%20Git/8.0%20Git底层原理/内存模型/对象/Git%20Object对象.md)
### 2 方案设计

#### 2.1 核心功能
#### 2.1.1 压缩和hash文件
- 1）初始化
	- 临时文件:当前临时文件不支持多次操作，后面会优化
	- 获取元数据主要是文件大小，blob文件需要使用
	- 构建压缩写入器，将压缩内容写入文件，使用(设计模式-装饰器模式)[1.2 Rust中的IO装饰器，流式写入同步计算设计](../../../../Areas/basic/系统设计和架构/Area/设计模式/23经典设计模式/结构型/装饰器模式/1.2%20Rust中的IO装饰器，流式写入同步计算设计.md)
- 2) 执行
	- 写入文件头部分
	- 写入压缩后的内容
	- 完成压缩，写入压缩要求的结尾格式和执行hash
- 3）结果处理
	- 根据bool write 来执行是否写入还是不写入`.git/objects`中
#### 2.1.2 读取object对象内容
- 1）解压缩读取mode和size信息
	- 使用buf包装ZlibDecoder，使用ZlibDecoder包装f
	- 读取c风格字符串
	- 转换为rust风格
	- 读取头信息
		- 使用枚举匹配类型
		- 使用parse解析数字
- 2）剩余内容打印到屏幕

### 3. 实现

#### 实现 hash_object
```rust
/// 计算 Git 对象 SHA-1，并返回压缩后的内容
///
/// # 参数
/// - `path`: 文件路径，支持任何 AsRef<Path>
///
/// # 返回
/// - Ok(sha1_hex):
///   - sha1_hex: 对象 SHA-1 的 16 进制表示
///
/// # 错误
/// - 读取文件或压缩失败会返回 Err(std::io::Error)
async fn hash_and_compress_file(path: &PathBuf, write: bool) -> Result<String, anyhow::Error> {
    // 使用tempfile crate创建临时文件
    let temp_file = NamedTempFile::new()?;
    let temp_path = temp_file.path().to_path_buf();

    // 获取文件元数据
    let stat = fs::metadata(path)
        .await
        .with_context(|| format!("stat file metadata failed: {}", path.display()))?;

    // 构建writer ，writer 是一个压缩写入器，压缩后的内容会写入到临时文件中
    let writer = ZlibEncoder::new(temp_file, Compression::default());
    // 使用HashWriter 包装writer，HashWriter 会计算写入的内容的hash
    let mut writer = HashWriter {
        writer,
        hasher: Sha1::new(),
    };

    // 1. 构造 Git blob header 并计算 SHA-1
    write!(writer, "blob {}\0", stat.len())?;

    // 2. 读取文件原始内容
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    std::io::copy(&mut file, &mut writer).context("stream file into blob")?;

    // 3. 计算hash和压缩，hash是和压缩一起进行的
    let _ = writer.writer.finish()?;
    let hex_sha1 = hex::encode(writer.hasher.finalize());

    // 4. 为什么需要临时文件，因为压缩后的文件名称和地址是根据 hash
    // 计算的，所以需要先压缩到临时文件，
    if write {
        fs::create_dir_all(format!(".git/objects/{}/", &hex_sha1[..2])).await?;
        std::fs::rename(
            temp_path,
            format!(".git/objects/{}/{}", &hex_sha1[..2], &hex_sha1[2..]),
        )
        .context("move blob file into .git/objects")?;
    }

    Ok(hex_sha1)
}
```
#### 读取文件
```rust
async fn decompress_file<P: AsRef<Path>>(path: P) -> Result<(), anyhow::Error> {
    let f = std::fs::File::open(path).context("open in .git/objects")?;
    let decoder = ZlibDecoder::new(f);
    // 使用buf包装了
    let mut buf = std::io::BufReader::new(decoder);
	// 1. 读取文件头
    let mut ret = Vec::new();
    buf.read_until(b'\0', &mut ret)?;
    // let s = std::str::from_utf8(&ret).unwrap();
    let c_str = CStr::from_bytes_with_nul(&ret).expect("Invalid C string");
    let header = c_str
        .to_str()
        .context(" .git/objects file header isn't valid utf-8")?;
    // 使用split_once 而不是split 是为了避免文件名中包含空格
    let Some((kind, size)) = header.split_once(' ') else {
        anyhow::bail!(".git/objects file header did not start with a konw type {header}");
    };
    // 处理类型
    let kind = match kind {
        "blob" => Kind::Blob,
        _ => anyhow::bail!("we do not know how to print a '{kind}'"),
    };

    // 要得到 usize，必须显式解析：
    let size = size
        .parse::<u64>()
        .context(" .git/objects file header size isn't valid:{size}")?;
    let mut buf = buf.take(size);
    let mut stdout = std::io::stdout().lock();
    // 直接使用std::io::copy
    match kind {
        Kind::Blob => {
            let n = std::io::copy(&mut buf, &mut stdout)?;
            anyhow::ensure!(
                n == size,
                ".git/objects file was not be expected size (expected {size}, got {n})"
            );
        }
        _ => anyhow::bail!("we do not know how to print a '{kind:?}'"),
    };
    Ok(())
}

```
### 4. 测试
- git初始化，使用脚本判断是否正确初始化，具体见：.test/test_cat_file.sh
- 写入测试文件，计算文件哈希，验证文件内容与预期一致，具体见：.test/test_cat_file.sh

### 5.总结
- 如果你新增了一个空提交，可以引用相同的树
- 如果你新增一个树，里面的对象只有一个做了修改，其他引用相同的对象key


