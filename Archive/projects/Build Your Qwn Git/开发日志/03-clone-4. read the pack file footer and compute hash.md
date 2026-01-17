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
### 2.1 核心实现
```rust
pub struct HashReader<R> {

    inner: R,

    hasher: Sha1,

}

impl<R: BufRead> HashReader<R> {

    pub fn new(inner: R) -> Self {

        Self {

            inner,

            hasher: Sha1::new(),

        }

    }

  

    pub fn finalize(self) -> ([u8; 20], R) {

        (self.hasher.finalize().into(), self.inner)

    }

}

impl<R: Read> Read for HashReader<R> {

    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {

        let n = self.inner.read(buf)?;

        self.hasher.update(&buf[..n]);

        Ok(n)

    }

}

impl<R: BufRead> BufRead for HashReader<R>

where

    R: BufRead,

{

    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {

        self.inner.fill_buf()

    }

  

    fn consume(&mut self, amt: usize) {

        let buf = &self.inner.fill_buf().unwrap()[..amt];

        self.hasher.update(buf); // ✅ 在 consume 时更新哈希

        self.inner.consume(amt);

    }

}
```

装饰器模式（Decorator Pattern）是一种 **结构型设计模式**，核心思想是：
1. 在不修改原有对象接口的前提下，动态地给对象添加功能。
2. 装饰器与被装饰对象共享相同接口，通常通过组合（containment）实现。

### 2.2 验证
- 读取最后的hash与计算的hash验证
```rust
let (hash, mut bufreader) = bufreader.finalize();

    eprintln!("final packfile hash: {}", hex::encode(hash));

  

    let mut expect_hash = Vec::new();

    bufreader

        .read_to_end(&mut expect_hash)

        .context("read final hash fail")?;

    eprintln!("expected packfile hash: {}", hex::encode(&expect_hash));
```
#### 分析 `HashReader`
1. **组合**：
    - `HashReader` 内部持有 `inner: R`，这个 `R` 可以是任何实现了 `BufRead` 或 `Read` 的类型。
2. **接口透明**：
    - `HashReader` 自身实现了 `Read` 和 `BufRead`，接口和 `R` 一致，可以像操作普通 reader 一样使用。
3. **附加功能**：
    - 额外增加了 SHA-1 哈希功能，在读取数据时自动更新哈希，而不影响原始 reader 的行为。

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