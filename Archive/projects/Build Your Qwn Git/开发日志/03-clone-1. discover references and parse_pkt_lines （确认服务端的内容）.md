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
### 2 方案设计
#### 2.1 构造和发送请求
```rust
    let repo_url = repo_url.trim_end_matches(".git").trim_end_matches('/');

    let client = reqwest::Client::new();

    let git_url = format!("{}/info/refs?service=git-upload-pack", repo_url);

    let resp = client.get(&git_url).send().await?;
```
#### 2.2 校验
##### 2.2.1 验证状态码
```rust
/// 校验 Git Upload-Pack 的 HTTP 状态码

pub async fn validate_status_and_return_body(

    resp: reqwest::Response,

    url: &str,

) -> Result<Bytes, anyhow::Error> {

    let status: StatusCode = resp.status();

    eprintln!("url: {}, status: {}", url, status);

    match status {

        StatusCode::OK => Ok(resp.bytes().await?),

        StatusCode::NOT_FOUND => {

            bail!("repository not found");

        }

  

        _ => {

            bail!("clone failed");

        }

    }

}
```
- 校验状态码Bytes
- 这里只得一提的是[Rust-Bytes-基本概念](../../../../Areas/Rust/Area/3%20库/库/crate/reqwest/Rust-Bytes-基本概念.md)Byte可以直接当成切片用

##### 2.2.2 客户端必须验证响应实体的前五个字节是否与正则表达式

```rust
fn validate_response(body: &[u8]) -> Result<(), anyhow::Error> {

    // 读取前五个字节并转为字符串

    let first_five_str = str::from_utf8(

        body.get(..5)

            .ok_or_else(|| anyhow::anyhow!("Response too short"))?,

    )?;

    anyhow::ensure!(

        RESPONSE_RE.is_match(first_five_str),

        "Response validation failed"

    );

    Ok(())
}
```
- 我们需要一个静态不可变变量，为什么不用常量，因为RESPONSE_RE正则表达式是运行时生成的
```rust
static RESPONSE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[0-9a-f]{4}#").unwrap());
```
#### 2.3 客户端必须将整个响应解析为一系列 pkt-line 记录

```rust
fn parse_pkt_lines(mut body: &[u8]) -> Result<Vec<String>, anyhow::Error> {

    let mut pkt_lines = Vec::new();

  

    while !body.is_empty() {

        // 1.读取前 4 个字节作为长度

        let (len_bytes, rest) = body.split_first_chunk::<4>().ok_or_else(|| {

            anyhow::anyhow!(

                "Incomplete pkt-line header: expected 4 bytes for length, got {}",

                body.len()

            )

        })?;

  

        // 2. 解析十六进制长度（Git pkt-line 使用十六进制，不是十进制！）

        let len_str =

            str::from_utf8(len_bytes).context("Invalid UTF-8 in pkt-line length header")?;

  

        let len = usize::from_str_radix(len_str, 16)

            .with_context(|| format!("Invalid hexadecimal pkt-line length: '{}'", len_str))?;

  

        eprintln!("pkt-line length: {}", len);

        // 3. 处理 flush packet (长度为0)

        if len == 0 {

            body = rest;

            continue;

        }

  
        // 4. 严格的长度校验
        anyhow::ensure!(

            len >= 4,

            "Invalid pkt-line length: {} (minimum valid length is 4)",

            len

        );

        anyhow::ensure!(

            body.len() >= len,

            "Insufficient data for pkt-line: need {} bytes, have {}",

            len,

            body.len()

        );

        // 5. 解析内容为 UTF-8 字符串

        // 提取 pkt-line 内容（不包含长度字段）

        let content_str = str::from_utf8(

            body.get(4..len)

                .ok_or_else(|| anyhow::anyhow!("Invalid pkt-line content"))?,

        )

        .context("Invalid UTF-8 in pkt-line content")?;

        pkt_lines.push(content_str.to_string());

        eprint!("pkt-line content: {}", content_str);

        body = &body[len..];

    }

  

    Ok(pkt_lines)

}

```
- rust
	- [Rust-slice 切片-split_first_chunk-固定长度切割切片](../../../../Areas/Rust/Area/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-split_first_chunk-固定长度切割切片.md)
	- [Rust-num-u16-from_str_radix](../../../../Areas/Rust/Area/3%20库/库/标准库/core/num/Rust-num-u16-from_str_radix.md)
- 业务逻辑
	- [Git-gitprotocol-pack - 0-如何通过网络传输软件包](../notes/Git-智能协议/Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)


### 3. 核心实现逻辑
- 去使用智能协议去获取引用列表，此操作会确定服务器拥有而客户端没有的数据，之后会发送git-upload-pack获取数据做准备
- 主要参考：[Git-gitprotocol-pack - 0-如何通过网络传输软件包](../notes/Git-智能协议/Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)
### 4. 测试
- [智能协议测试](../notes/Git-智能协议/智能协议测试.md)
### 5.总结
