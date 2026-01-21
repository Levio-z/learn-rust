### 1 参考
#### 1.1 挑战内容
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-04-kp1.md
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-05-fe4.md
- https://github.com/codecrafters-io/build-your-own-git/blob/main/stage_descriptions/base-06-jm9.md
#### 1.2 资料

[Git对象](https://git-scm.com/book/zh/v2/Git-%e5%86%85%e9%83%a8%e5%8e%9f%e7%90%86-Git-%e5%af%b9%e8%b1%a1)
[Git引用](https://git-scm.com/book/zh/v2/Git-%e5%86%85%e9%83%a8%e5%8e%9f%e7%90%86-Git-%e5%bc%95%e7%94%a8)

#### 1.3 知识点
- 文件相关：[rust-std-fs-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/fs/rust-std-fs-TOC.md)
- zlibDecoder：[ZlibEncoder-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/crate/flate2/struct/ZlibEncoder/ZlibEncoder-TOC.md)
- 什么是c风格字符串：[Rust  CStr TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/ffi/Cstr/Rust%20%20CStr%20TOC.md)
- 分割字符串：[Rust `&str` 常用字符串操作方法概览](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/core/str/常用方法/Rust%20`&str`%20常用字符串操作方法概览.md)
- 装饰器（限制读取）：[take](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/io/BufRead/take.md)
- 解析为数值：[解析为数值](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/core/str/常用方法/解析为数值.md)
- print的底层：[rust-stdout-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/io/stdout/rust-stdout-TOC.md)
	- [copy reader to writer](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/io/methods/io%20copy/copy%20reader%20to%20writer.md)
	- [writeln! 与write_all](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/io/stdout/研究/writeln!%20与write_all.md)
- 错误处理： [rust-anyhow-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/crate/anyhow/rust-anyhow-TOC.md)
- 临时文件：[rust-crate-tempfile-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/crate/tempfile/rust-crate-tempfile-TOC.md)
- 命令行工具：[rust-clap-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/clap/rust-clap-TOC.md)
- 文件格式和基本流程梳理(建议单独开启一个网址来参考)：[Git Object对象](../../../../Areas/basic/Git/Pro%20Git/8.0%20Git底层原理/内存模型/对象/Git%20Object对象.md)
- 闭包：[rust-闭包-TOC](../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.4%20函数式编程特性/2.4.1%20闭包/rust-闭包-TOC.md)
- 迭代器：[rust-iter-TOC](../../../../Areas/Rust/Area/3%20库&应用场景/库/标准库/std/iter/rust-iter-TOC.md)
### 2 方案设计
#### 2.1 数据结构设计
##### 2.1.1 数据结构设计
**类型枚举**
```rust
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Blob,
    Tree,
    Commit,
    Tag,
}
```
**可读取对象**
```rust
#[derive(Debug)]
pub(crate) struct Object<R> {
    pub(crate) kind: Kind,
    pub(crate) expected_size: u64,
    pub(crate) reader: R,
}
```
**tree中项目的模式**
```rust
#[derive(Debug, PartialEq, Eq)]

pub(crate) enum Mode {

    File,

    Executable,

    Directory,

    SymbolicLink,

}
```
#### 2.2 重构：封装逻辑，提高代码复用

##### 2.2.1 文件名->object：传入hash从 Git 对象数据库读取某个对象
从 Git 对象数据库读取某个对象，并返回一个带类型信息和大小限制的 reader，这在读取tree和blob文件都会使用
```rust
pub(crate) async fn hash_to_reader(path: &str) -> anyhow::Result<Object<impl BufRead>> {

    // TODO tokio-util + async-compression 提供异步 Zlib 解压：

    // 使用string构造路径

    let f = std::fs::File::open(format!(".git/objects/{}/{}", &path[0..2], &path[2..]))

        .context("open in .git/objects")?;

    let decoder = ZlibDecoder::new(f);

    let mut buf = std::io::BufReader::new(decoder);

  

    let mut ret = Vec::new();

    // 1. 读取文件头

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

        "tree" => Kind::Tree,

        "commit" => Kind::Commit,

        "tag" => Kind::Tag,

        _ => anyhow::bail!("we do not know how to print a '{kind}'"),

    };

  

    // 要得到 usize，必须显式解析：

    let size = size

        .parse::<u64>()

        .context(" .git/objects file header size isn't valid:{size}")?;

    let buf = buf.take(size);

    Ok(Object {

        kind,

        expected_size: size,

        reader: buf,

    })

}
```
##### 2.2.2 压缩和写入：执行文件压缩和hash
###### 装饰器模式：数据结构设计
- 根据bool去判断是否需要hash，使用装饰器模式

```rust
/// 包装器，根据 compress 决定是否压缩

enum MaybeCompress<W: Write> {

    Compressed(ZlibEncoder<W>),

    Plain(W),

}

  

impl<W: Write> Write for MaybeCompress<W> {

    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {

        match self {

            MaybeCompress::Compressed(z) => z.write(buf),

            MaybeCompress::Plain(w) => w.write(buf),

        }

    }

  

    fn flush(&mut self) -> std::io::Result<()> {

        match self {

            MaybeCompress::Compressed(z) => z.flush(),

            MaybeCompress::Plain(w) => w.flush(),

        }

    }

}

  

impl<W: Write> MaybeCompress<W> {

    fn finish(&mut self) -> std::io::Result<()> {

        match self {

            MaybeCompress::Compressed(z) => z.try_finish(), // 完成压缩

            MaybeCompress::Plain(_) => Ok(()),

        }

    }

}
```

###### 计算hash和压缩逻辑
```rust
/// 压缩hash文件的方法：根据传入条件来判断是否压缩

    pub(crate) async fn compute_hash(

        &mut self,

        writer: impl Write,

        compress: bool,

    ) -> Result<[u8; 20], anyhow::Error> {

        // 1、根据compress是否要压缩，包装writer

        let writer = if compress {

            MaybeCompress::Compressed(ZlibEncoder::new(writer, Compression::default()))

        } else {

            MaybeCompress::Plain(writer)

        };

  

        // 2、使用HashWriter 包装writer，HashWriter 会计算写入的内容的hash

        let mut writer = HashWriter {

            writer,

            hasher: Sha1::new(),

        };

        write!(writer, "{} {}\0", self.kind, self.expected_size)?;

        // 3、将reader 中的内容写入writer

        std::io::copy(&mut self.reader, &mut writer).context("stream file into blob")?;

  

        // 4. 计算hash和压缩，hash是和压缩一起进行的

        let _ = writer.writer.finish()?;

        let sha1 = writer.hasher.finalize();

        Ok(sha1.into())

    }

```
##### 2.2.3 Object写入文件：将压缩后的对象写入 .git/objects 目录
```rust
    pub(crate) async fn write_object(&mut self) -> Result<[u8; 20], anyhow::Error> {

        // 1、使用tempfile crate创建临时文件

        let tmp_path = NamedTempFile::new()?.into_temp_path();

        let file: std::fs::File = std::fs::File::create(&tmp_path)?;

  

        // 2、计算hash 压缩写入临时文件

        let hex_sha1 = self

            .compute_hash(file, true)

            .await

            .context("compute hash failed")?;

        let hex = hex::encode(hex_sha1);

  

        // 3、重命名文件，将临时文件重命名为最终的文件

        fs::create_dir_all(format!(".git/objects/{}/", &hex[..2])).await?;

        std::fs::rename(

            tmp_path,

            format!(".git/objects/{}/{}", &hex[..2], &hex[2..]),

        )

        .context("move blob file into .git/objects")?;

  

        Ok(hex_sha1)

    }
```
##### 2.2.4 文件->object：读取文件内容转换为object对象
```rust
pub(crate) fn file_to_object(file: impl AsRef<Path>) -> anyhow::Result<Object<impl Read>> {

    let file = file.as_ref();

    let stat = std::fs::metadata(file).with_context(|| format!("stat {}", file.display()))?;

    // TODO: technically there's a race here if the file changes between stat and

    // write

    let file = std::fs::File::open(file).with_context(|| format!("open {}", file.display()))?;

    Ok(Object {

        kind: Kind::Blob,

        expected_size: stat.len(),

        reader: file,

    })

}
```


#### 2.2 ls-tree命令实现
- 1）调用读取对象逻辑
- 2)  循环解析mode和name然后根据是否有name_only切换不同输出方式
	- .split_once(' ')分割&str为两个&str
	- 知道hash长度，使用read_exact精确读取
	- 根据不同输出策略格式化输出内容
		- 为了让kind从mode转换更方便，为kind实现from方法

#### 2.3 write_tree命令实现
- 1）遍历当前目录文件夹，初始化vec
	- 构造vec，提取后续需要的信息
		- mode 写入内容
		- name 写入内容
		- path 文件地址，写入blob
- 2）排序
- 3）迭代处理目录和文件
	- 硬编码排除.git
	- 获取hash
		- 文件
			- 使用file_to_object读取object对象，使用write_object写入
		- 目录
			- 递归调用invoke处理
	- 按照格式写入不同的项
#### 2.4 commit命令实现
1）读取当前工作区指向的引用：`.git/HEAD` 是一个特殊文件，用于记录「当前工作区指向的引用」, 通常指向当前所在分支（如内容为 `ref: refs/heads/main`，表示当前在 `main` 分支）；
2）确定当前分支的「上一次提交哈希」（作为新提交的父提交）。
3）调用write-tree 生成树对象（tree object）哈希
- 树对象是 Git 中用于记录「目录结构和文件内容快照」的核心数据结构，它会递归记录所有文件的名称、权限和内容哈希。
4）生成commit对象
- 按照格式写入文件内容
5）更新分支引用，完成提交

### 3. 核心实现逻辑

#### 实现 ls-tree
- 复用文件名->object逻辑

```rust
pub(crate) async fn invoke(path: &str, name_only: bool) -> Result<(), anyhow::Error> {
	// 文件名->object
    let mut hash_object = hash_to_reader(path).await?;

    // 直接使用std::io::copy将内容输出到终端

    match hash_object.kind {

        Kind::Tree => {

            let mut buf = Vec::new();

            let stdout = std::io::stdout();

            // 自带缓冲

            let mut stdout = stdout.lock();

            let mut hashbuf = [0; 20];

            loop {

                let n = hash_object

                    .reader

                    .read_until(0, &mut buf)

                    .context("read next tree object entry")?;

                if n == 0 {

                    break;

                }

                let mode_and_name = CStr::from_bytes_with_nul(&buf)

                    .context("invalid tree entry")?

                    .to_str()

                    .context("invalid tree entry")?;

                // split_once https://github.com/rust-lang/rust/issues/112811

                let (mode, name) = mode_and_name

                    .split_once(' ')

                    .context("split always yields once")?;

  

                hash_object.reader.read_exact(&mut hashbuf).context("read entry hash fail")?;

                if name_only {

                    writeln!(&mut stdout, "{name}")?;

                } else {

                    let kind: Kind = Mode::from_str(mode)?.into();

                    writeln!(

                        &mut stdout,

                        "{mode:0>6} {} {}  {name}",

                        kind,

                        hex::encode(hashbuf),

                    )?;

                }

                buf.clear();

            }

        }

        _ => anyhow::bail!("we do not know how to print a '{:?}'", hash_object.kind),

    };

    Ok(())

}
```
#### 实现write_tree
**invoke**
- 构造tree文件内容写入文件
```rust
pub(crate) async fn invoke(path: PathBuf) -> Result<[u8; 20], anyhow::Error> {

    Ok(write_tree(path)

        .await

        .context("invoke write tree failed")?

        .as_mut()

        .ok_or_else(|| anyhow::anyhow!("invoke write tree failed"))?

        .write_object()

        .await

        .context("invoke write tree failed")?)

}
```

**write_tree**
- 构造tree方法
```rust
pub(crate) fn write_tree(path: PathBuf) -> TreeFuture {

    Box::pin(async move {

        let mut dir = fs::read_dir(path).await.context("open directory failed")?;

        let mut vec = Vec::new();

  

        while let Some(entry) = dir.next_entry().await.context("read directory failed")? {

            let name = entry.file_name();

            let path = entry.path();

            let mode = Mode::from_path(&path).await?;

            vec.push((name, path, mode));

        }

  

        vec.sort_by(|a, b| {

            let afn = a.0.as_encoded_bytes();

            let bfn = b.0.as_encoded_bytes();

  

            let prefix_cmp = afn

                .iter()

                .zip(bfn.iter())

                .find_map(|(x, y)| if x != y { Some(x.cmp(y)) } else { None });

  

            if let Some(ord) = prefix_cmp {

                return ord;

            }

  

            let common_len = afn.len().min(bfn.len());

            let next_byte_or_slash = |bytes: &[u8], len: usize, is_dir: bool| {

                bytes

                    .get(len)

                    .copied()

                    .or(if is_dir { Some(b'/') } else { None })

            };

            let a_next = next_byte_or_slash(afn, common_len, a.2.is_dir());

            let b_next = next_byte_or_slash(bfn, common_len, b.2.is_dir());

  

            a_next.cmp(&b_next)

        });

  

        let mut tree_object = Vec::new();

        for item in vec {

            let hash = if Mode::Directory == item.2 {

                if item.0 == ".git" {

                    continue;

                }

                invoke(item.1).await?

            } else {

                crate::objects::file_to_object(&item.1)?

                    .write_object()

                    .await

                    .context("write object failed")?

            };

            tree_object.write_all(item.2.to_bytes())?;

            tree_object.write_all(b" ")?;

            tree_object.write_all(item.0.as_encoded_bytes())?;

            tree_object.write_all(b"\0")?;

            tree_object.write_all(&hash)?;

        }

  

        if tree_object.is_empty() {

            Ok(None)

        } else {

            Ok(Some(Object {

                kind: Kind::Tree,

                expected_size: tree_object.len() as u64,

                reader: Cursor::new(tree_object),

            }))

        }

    })

}

```
#### 实现commit
```rust
pub(crate) async fn invoke_commit_tree(

    tree_sha: String,

    message: String,

    parent: Option<String>,

) -> Result<[u8; 20], anyhow::Error> {

    let mut buf = Vec::new();

    writeln!(buf, "tree {tree_sha}")?;

    if let Some(parent) = parent {

        writeln!(buf, "parent {parent}")?;

    }

    let name = "Levio-Z";

    let email = "67247011+Levio-z@users.noreply.github.com";

    let time = std::time::SystemTime::now()

        .duration_since(std::time::SystemTime::UNIX_EPOCH)

        .context("current system time is before UNIX epoch")?

        .as_secs();

    writeln!(buf, "author {name} <{email}> {time} +0800")?;

    writeln!(buf, "committer {name} <{email}> {time} +0800")?;

    writeln!(buf)?;

    writeln!(buf, "{message}")?;

  

    let mut commit = Object {

        kind: Kind::Commit,

        expected_size: buf.len() as u64,

        reader: Cursor::new(buf),

    };

    let hash = commit.write_object().await?;

    println!("{}", hex::encode(hash));

    Ok(hash)

}

  

pub(crate) async fn invoke_commit(message: String) -> Result<(), anyhow::Error> {

    let head_ref = std::fs::read_to_string(".git/HEAD").context("read HEAD")?;

    let Some(head_ref) = head_ref.strip_prefix("ref: ") else {

        anyhow::bail!("refusing to commit onto detached HEAD");

    };

    // 去除末尾的换行符

    let head_ref = head_ref.trim_end();

    let parent = if let Ok(hash) = std::fs::read_to_string(format!(".git/{head_ref}")) {

        Some(hash.trim().to_string())

    } else {

        None

    };

  

    // 计算hash

    let tree_hash = crate::commands::write_tree::invoke(PathBuf::from("."))

        .await

        .context("write tree")?;

  

    // 提交hash

    let commit_hash = invoke_commit_tree(hex::encode(tree_hash), message, parent)

        .await

        .context("commit tree")?;

    let commit_hash = hex::encode(commit_hash);

    std::fs::write(format!(".git/{head_ref}"), &commit_hash)

        .with_context(|| format!("update HEAD reference target {head_ref}"))?;

    println!("HEAD is now at {commit_hash}");

    Ok(())

}
```

### 4. 测试
- 测试程序自动测试：使用我们的程序初始化并生成tree，使用官方git生成tree，比较ls-tree的内容是否一致具体见.test/test_ls_tree.sh
- 硬编码好提交的个人信息，使用官方git init 然后修改文件，使用程序完成提交，然后git log查看提交历史，push到GitHub，观察是否正常

### 5.总结
- 实现ls-tree不难，可以复用read a blob的逻辑
- 实现write-tree，难点在于设计递归以及比较代码：迭代器和闭包的应用
- write_all写入字节的时候不要放在一排，多拍单独写对性能影响不大，但对可读性提升


