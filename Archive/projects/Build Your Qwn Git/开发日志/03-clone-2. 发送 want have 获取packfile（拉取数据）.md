### 1 参考
#### 1.1 挑战内容
- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/discovering-references
- 使用智能协议请求仓库将内容下载下来
#### 1.2 资料
- [This forum post](https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/intro)
	- 如何逐步实现这一点的建议
- Git 的[智能 HTTP 传输协议](https://www.git-scm.com/docs/http-protocol)
- [Git-gitprotocol-pack - 0-如何通过网络传输软件包](../notes/Git-智能协议/Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)
#### 1.3 知识点
- 文件相关：[rust-std-fs-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/fs/rust-std-fs-TOC.md)
- zlibDecoder：[ZlibEncoder-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/flate2/struct/ZlibEncoder/ZlibEncoder-TOC.md)
- 什么是c风格字符串：[Rust  CStr TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/ffi/Cstr/Rust%20%20CStr%20TOC.md)
- 分割字符串：[Rust `&str` 常用字符串操作方法概览](../../../../Areas/Rust/Area/3%20库/库/标准库/core/str/常用方法/Rust%20`&str`%20常用字符串操作方法概览.md)
- 装饰器（限制读取）：[take](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/BufRead/take.md)
- 解析为数值：[解析为数值](../../../../Areas/Rust/Area/3%20库/库/标准库/core/str/常用方法/解析为数值.md)
- print的底层：[rust-stdout-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/stdout/rust-stdout-TOC.md)
	- [copy reader to writer](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/methods/io%20copy/copy%20reader%20to%20writer.md)
	- [writeln! 与write_all](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/stdout/研究/writeln!%20与write_all.md)
- 错误处理： [rust-anyhow-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/anyhow/rust-anyhow-TOC.md)
- 临时文件：[rust-crate-tempfile-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/tempfile/rust-crate-tempfile-TOC.md)
- 命令行工具：[rust-clap-TOC](../../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/clap/rust-clap-TOC.md)
- 文件格式和基本流程梳理(建议单独开启一个网址来参考)：[Git Object对象](../../../../Areas/basic/Git/Pro%20Git/8.0%20Git底层原理/内存模型/对象/Git%20Object对象.md)
- 闭包：[rust-闭包-TOC](../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.4%20函数式编程特性/2.4.1%20闭包/rust-闭包-TOC.md)
- 迭代器：[rust-iter-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/iter/rust-iter-TOC.md)
- 字符串去除尾部：[Rust-String-去除结尾特定内容的方法-toc](../../../../Areas/Rust/Area/3%20库/库/标准库/alloc/String/methods/Rust-String-去除结尾特定内容的方法-toc.md)
- reqwest：[Rust-reqwest-基本概念-TOC](../../../../Areas/Rust/Area/3%20库/库/crate/reqwest/Rust-reqwest-基本概念-TOC.md)
	- [Rust-Bytes-基本概念](../../../../Areas/Rust/Area/3%20库/库/crate/reqwest/Rust-Bytes-基本概念.md)
- 切片和数组的git方法：
	- [Rust-slice 切片-get(..)](../../../../Areas/Rust/Area/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-get(..).md)
	- [Rust-slice 切片-split_first_chunk-固定长度切割切片](../../../../Areas/Rust/Area/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-split_first_chunk-固定长度切割切片.md)
- 链式处理：[Rust-Result-and_then](../../../../Areas/Rust/Area/3%20库/库/标准库/std/option/Option/methods/Rust-Result-and_then.md)
- Cursor：[Rust-cursor-基本概念-TOC](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/Cursor/Rust-cursor-基本概念-TOC.md)
- 缓冲读：[Rust-io-BufReader](../../../../Areas/Rust/Area/3%20库/库/标准库/std/io/BufRead/Rust-io-BufReader.md)
### 2 方案设计
#### 2.1 解析commit hash和分支名
##### 2.1.1 解析commit hash
```rust
    let (hash, rest) = line

        .split_once(' ')

        .context("failed to split pkt-line into <hash> <capabilities>")?;

    eprintln!("first hash: {}", hash);
```
##### 2.1.2 解析分支名

```rust
let symref_prefix = "symref=HEAD:";

    let branch = rest

        .split_once(symref_prefix)

        .ok_or_else(|| anyhow::anyhow!("missing `symref=HEAD:` capability"))

        .and_then(|(_, r)| {

            r.split_whitespace()

                .next()

                .ok_or_else(|| anyhow::anyhow!("missing branch after `symref=HEAD:`"))

        })?;

    eprintln!("branch: {}", branch);
```

#### 2.2 客户端发送协商请求需要的数据

```rust
    let clone_url = format!("{}/git-upload-pack", repo_url);

  

    let mut req_body = Vec::new();

    writeln!(req_body, "0032want {}", hash)?;

    writeln!(req_body, "00000009done")?;

  

    let commit_hash = hex::decode(hash)?;

    let resp = client

        .post(&clone_url)

        .header("Content-Type", "application/x-git-upload-pack-request")

        .body(req_body)

        .send()

        .await?;

    let mut body = Cursor::new(validate_status_and_return_body(resp, &clone_url).await?);

    // &[u8] 本身实现了 Read：

    let mut bufreader = std::io::BufReader::new(&mut body);
```

### 3. 核心实现逻辑
- 解析引用列表中的head指向的commit的hash和分支，随后构造want have请求packfile
- 主要参考：[[协商阶段]]
### 4. 测试
- [智能协议测试](../notes/Git-智能协议/智能协议测试.md)
### 5.总结
