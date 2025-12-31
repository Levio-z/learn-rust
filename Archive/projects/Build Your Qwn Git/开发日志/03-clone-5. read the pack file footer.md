### 1 参考
#### 1.1 挑战内容
- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/discovering-references
- 使用智能协议请求仓库将内容下载下来
#### 1.2 资料
- [This forum post](https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/intro)
	- 如何逐步实现这一点的建议
- Git 的[智能 HTTP 传输协议](https://www.git-scm.com/docs/http-protocol)
- [Git-gitprotocol-pack - 0-如何通过网络传输软件包](../notes/Git-智能协议/Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)
	- [Git-gitprotocol-pack - 3-如何通过网络传输软件包-packfile文件格式](../notes/Git-智能协议/Git-gitprotocol-pack%20-%203-如何通过网络传输软件包-packfile文件格式.md)
#### 1.3 知识点
- 文件相关：[rust-std-fs-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/fs/rust-std-fs-TOC.md)
- zlibDecoder：[ZlibEncoder-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/flate2/struct/ZlibEncoder/ZlibEncoder-TOC.md)
- 什么是c风格字符串：[Rust  CStr TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/ffi/Cstr/Rust%20%20CStr%20TOC.md)
- 分割字符串：[Rust `&str` 常用字符串操作方法概览](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/core/str/常用方法/Rust%20`&str`%20常用字符串操作方法概览.md)
- 装饰器（限制读取）：[take](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/BufRead/take.md)
- 解析为数值：[解析为数值](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/core/str/常用方法/解析为数值.md)
- print的底层：[rust-stdout-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/stdout/rust-stdout-TOC.md)
	- [copy reader to writer](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/methods/io%20copy/copy%20reader%20to%20writer.md)
	- [writeln! 与write_all](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/stdout/研究/writeln!%20与write_all.md)
- 错误处理： [rust-anyhow-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/anyhow/rust-anyhow-TOC.md)
- 临时文件：[rust-crate-tempfile-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/tempfile/rust-crate-tempfile-TOC.md)
- 命令行工具：[rust-clap-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/clap/rust-clap-TOC.md)
- 文件格式和基本流程梳理(建议单独开启一个网址来参考)：[Git Object对象](../../../../Areas/basic/Git/Pro%20Git/8.0%20Git底层原理/内存模型/对象/Git%20Object对象.md)
- 闭包：[rust-闭包-TOC](../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.4%20函数式编程特性/2.4.1%20闭包/rust-闭包-TOC.md)
- 迭代器：[rust-iter-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/iter/rust-iter-TOC.md)
- 字符串去除尾部：[Rust-String-去除结尾特定内容的方法-toc](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/alloc/String/methods/Rust-String-去除结尾特定内容的方法-toc.md)
- reqwest：[Rust-reqwest-基本概念-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/reqwest/Rust-reqwest-基本概念-TOC.md)
	- [Rust-Bytes-基本概念](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/reqwest/Rust-Bytes-基本概念.md)
- 切片和数组的git方法：
	- [Rust-slice 切片-get(..)](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-get(..).md)
	- [Rust-slice 切片-split_first_chunk-固定长度切割切片](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-split_first_chunk-固定长度切割切片.md)
- 链式处理：[Rust-Result-and_then](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/option/Option/methods/Rust-Result-and_then.md)
- Cursor：[Rust-cursor-基本概念-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/Cursor/Rust-cursor-基本概念-TOC.md)
- 缓冲读：[Rust-io-BufReader](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/BufRead/Rust-io-BufReader.md)
- vec：[Rust-vec-try_into-将vec转换为数组](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/alloc/Vec/Rust-vec-try_into-将vec转换为数组.md)
- fs
	- [Rust-fs-create_dir_all-递归创建目录](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/fs/Rust-fs-create_dir_all-递归创建目录.md)
	- [Rust-fs-write-写入切片作为整个文件内容](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/fs/Rust-fs-write-写入切片作为整个文件内容.md)
	- [Rust PathBuf](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/path/PathBuf/Rust%20PathBuf.md) 一个非常好用的有所有权可以增删的路径对象
### 2 方案设计
### 2.1 基本方法
### 2.1.1 从commit中解析出目录tree的hash
```rust
fn parse_tree_hash(

    commit_hash: &[u8],

    hashmap: &HashMap<[u8; 20], Object<Cursor<Vec<u8>>>>,

) -> anyhow::Result<[u8; 20]> {

    let mut head = hashmap

        .get(commit_hash)

        .context("can not find head object")?

        .clone();

    head.reader.set_position(5);

    let mut hash = [0; 40];

    head.reader.read_exact(&mut hash)?;

    let tree_hash = hex::decode(hash)?;

    tree_hash

        .try_into()

        .map_err(|v: Vec<u8>| anyhow::anyhow!("hash length invalid: {:?}", v))

}
```
- try_into()的err里面的内容是vec直接输出不太清晰，使用map_err转换错误

#### 2.1.2  写入文件
```rust
fn tree_to_file(

    path: PathBuf,

    tree_hash: &[u8],

    hashmap: &HashMap<[u8; 20], Object<Cursor<Vec<u8>>>>,

) -> anyhow::Result<()> {

    fs::create_dir_all(&path).context("create dir all fail")?;

    let mut hash_object = hashmap

        .get(tree_hash)

        .context("can not find tree object")?

        .clone();

    let mut buf = Vec::new();

    let mut hashbuf = [0; 20];

    loop {

        let n = hash_object

            .reader

            .read_until(0, &mut buf)

            .context("read next tree object entry")?;

        if n == 0 {

            break;

        }

        eprintln!("{:?}", String::from_utf8_lossy(&buf));

        let mode_and_name = CStr::from_bytes_with_nul(&buf)

            .context("invalid tree entry")?

            .to_str()

            .context("invalid tree entry")?;

        // split_once https://github.com/rust-lang/rust/issues/112811

        // mode 权限设置，非核心，暂时忽略

        let (mode, name) = mode_and_name

            .split_once(' ')

            .context("split always yields once")?;

  

        hash_object

            .reader

            .read_exact(&mut hashbuf)

            .context("read entry hash fail")?;

        let kind: Kind = Mode::from_str(mode)?.into();

        match kind {

            Kind::Tree => {

                tree_to_file(path.join(name), &hashbuf, hashmap)?;

            }

            Kind::Blob => {

                eprintln!("blob hash: {}", hex::encode(hashbuf));

                let blob_path = path.join(name);

                let content = &hashmap

                    .get(hashbuf.as_slice())

                    .context("can not find blob object")?

                    .reader;

                fs::write(blob_path, content.get_ref())?; // 自动创建文件并写入

            }

            _ => {}

        }

        buf.clear();

    }

    Ok(())

}
```

- 创建当前目录
	- fs::create_dir_all(&path).context("create dir all fail")?;
- 根据传入进来的treehash获取object对象，获取该目录下的内容
	- 如果是blob对象，直接创建新文件并写入
	- 如果是tree对象，也就是目录，递归调用自己

#### 2.1.3 初始化
- 创建git仓库的内容和head文件
```rust
pub(crate) async fn git_init(dir: PathBuf, mut branch: &str) -> anyhow::Result<()> {

    const GIT_DIR: &str = ".git";

  

    // 创建目录列表

    let dirs = ["objects", "refs"];

    fs::create_dir(dir.join(GIT_DIR))

        .await

        .context("create git dir fail")?;

    for d in dirs {

        fs::create_dir(dir.join(GIT_DIR).join(d))

            .await

            .with_context(|| format!("create git {d} dir fail"))?;

    }

    if branch.is_empty() {

        branch = "refs/heads/main";

    }

  

    // 创建 HEAD 文件

    fs::write(dir.join(GIT_DIR).join("HEAD"), format!("ref: {branch}\n"))

        .await

        .context("create git HEAD fail")?;

  

    Ok(())

}
```

#### 2.1.4 write_ref_file
- 写入文件，如果父目录不存在提前创建的封装
```rust
pub async fn write_ref_file(path: PathBuf, data: &[u8]) -> anyhow::Result<()> {

    // 1. 创建父目录

    if let Some(parent) = path.parent() {

        fs::create_dir_all(parent).await?;

    }

  

    // 2. 写入文件

    fs::write(path, hex::encode(data)).await?;

  

    Ok(())

}
```
### 2.2 实现
```rust
	let tree_hash = parse_tree_hash(&commit_hash, &hashmap)?;

    tree_to_file(path.to_path_buf(), &tree_hash, &hashmap)?;

    crate::objects::git_init(path.to_path_buf(), branch).await?;

    let head_ref = crate::objects::find_headref(path.to_path_buf())?;

    eprintln!("head ref: {}", head_ref);

    crate::objects::write_ref_file(path.join(format!(".git/{head_ref}")), &commit_hash)

        .await

        .context("write ref file fail")?;

    // 写入object文件

    for (_, mut object) in hashmap {

        object.write_object(path.to_path_buf()).await?;

    }
```
- parse_tree_hash：解析出treehash
- tree_to_file：将内容写入工作目录
- git_init：初始化仓库
- 找到head_ref：将commit_hash写入head_ref文件
- 将object对象写如object文件

### 3. 核心实现逻辑
- 解析引用列表中的head指向的commit的hash和分支，随后构造want have请求packfile
- 主要参考：
	- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/ref-delta
	- [Git-gitprotocol-pack - 3-如何通过网络传输软件包-packfile文件格式](../notes/Git-智能协议/Git-gitprotocol-pack%20-%203-如何通过网络传输软件包-packfile文件格式.md)
### 4. 测试
- [智能协议测试](../notes/Git-智能协议/智能协议测试.md)
### 5.总结
### 关联
- [1.0 装饰器模式](../../../../Areas/basic/系统设计和架构/Area/设计模式/23经典设计模式/结构型/装饰器模式/1.0%20装饰器模式.md)