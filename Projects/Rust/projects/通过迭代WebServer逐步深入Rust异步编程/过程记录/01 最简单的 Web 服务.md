### 建立一个TcpListener
```rust
use std::net::TcpListener;

fn main() {
    let listener = TcpListener::bind("localhost:3000").unwrap();
}

```
- 在指定地址和端口上创建一个监听 TCP 套接字（Listener），用于接收进入的连接
	- 调用成功后，可以使用该 `TcpListener` 实例调用：
		- `.accept()`：**接受传入连接，返回 `TcpStream`**
		- `.incoming()`：返回一个迭代器，可用于循环接收连接
		- `println!("Listening on {:?}", listener.local_addr()?);`
			- 打印地址
#### Web 服务相应来自客户端传入的连接请求，按顺序挨个执行它们。
```rust
use std::io;

use std::net::TcpListener;

use std::net::TcpStream;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let listener = TcpListener::bind("localhost:3000").unwrap();

    println!("Listening on {:?}", listener.local_addr()?);

  

    loop {

        let (connection, _) = listener.accept().unwrap();

        if let Err(e) = handle_connection(connection) {

            println!("failed to handle connection: {e}")

        }

    }

}

  

fn handle_connection(connection: TcpStream) -> io::Result<()> {

    // ...

}

```

TCP 连接是 Web 服务与客户端之间的双向数据通道，用 TcpStream 类型表示。它实现了 Read 和 Write 特性( trait )，抽象了 Tcp 内部细节，使我们能够读取或写入普通的字节数据。
- **`accept()` 会阻塞当前线程，直到有一个客户端发起 TCP 连接，并返回这个连接对应的 socket 对象（`TcpStream`）和客户端的地址。**
	- `accept()` 在 Rust 标准库 `TcpListener::accept()` 方法中是对操作系统的阻塞式系统调用的封装。
		- 检查监听套接字 `listener` 是否有新的连接请求；
			- 如果没有，就阻塞线程（除非设置了非阻塞）；
			- 如果有，就返回一个**新的 socket**（`TcpStream`）和**连接对方的地址**（`SocketAddr`）；
			- 这个新的 socket 是一个**可读写的“通信通道”**，你可以用它 `read()` 客户端发来的内容，也可以 `write()` 数据返回给客户端。
			- Result<(TcpStream, SocketAddr), std::io::Error>
#### 连接读取数据，写入缓存
```rust
fn handle_connection(connection: TcpStream) -> io::Result<()> {
    let mut request = [0u8; 1024];
    // ...

    Ok(())
}

```
接下来就是从连接读取数据，写入缓存。读取动作每次读到的字节数不定，所以需要跟踪已读取字节数。读取动作在循环中不断重复，依次将读到的内容写入缓存。
```rust
use std::io::Read;

fn handle_connection(mut connection: TcpStream) -> io::Result<()> {
    let mut read = 0;
    let mut request = [0u8; 1024];

    loop {
        // 尝试从流中读取数据
        let num_bytes = connection.read(&mut request[read..])?;

        // 跟踪已读取的字节数
        read += num_bytes;
    }

    Ok(())
}


```
直到读到一个特殊的字符序列 `\r\n\r\n`，我们_约定_用它来作为读取的结束标志。
```rust
fn handle_connection(mut connection: TcpStream) -> io::Result<()> {
    let mut read = 0;
    let mut request = [0u8; 1024];

    loop {
        // 尝试从流中读取数据
        let num_bytes = connection.read(&mut request[read..])?;

        // 跟踪已读取的字节数
        read += num_bytes;

        // 判断是否读到结束标记
        if request.get(read - 4..read) == Some(b"\r\n\r\n") {
            break;
        }
    }

    Ok(())
}
```

还有一种情况：如果连接已断开，读到的字节数将是0，此时如果客户端的请求未发送完毕，直接返回转入处理下一个连接。
再次声明，不用担心 HTTP 规范限制，我们的目的是让 Web 服务能工作就行。
```rust

fn handle_connection(mut connection: TcpStream) -> io::Result<()> {
    let mut read = 0;
    let mut request = [0u8; 1024];

    loop {
        // 尝试从流中读取数据
        let num_bytes = connection.read(&mut request[read..])?;

        // 客户端已断开
        if num_bytes == 0 { // 👈
            println!("client disconnected unexpectedly");
            return Ok(());
        }

        // 跟踪已读取的字节数
        read += num_bytes;

        // 判断是否读到结束标记
        if request.get(read - 4..read) == Some(b"\r\n\r\n") {
            break;
        }
    }

    Ok(())
}

```
一旦读取完成，就可以将读到的结果转换为字符串，并以日志形式输出到控制台。
```rust
fn handle_connection(stream: TcpStream) -> io::Result<()> {
    let mut read = 0;
    let mut request = [0u8; 1024];

    loop {
        // ...
    }

    let request = String::from_utf8_lossy(&request[..read]);
    println!("{request}");

    Ok(())
}

```
接下来实现响应部分
跟读操作类似，写操作也可能不会一次完成。所以同样需要采用循环的方式，不断写入响应数据到流中，每次写入从上一次结束的位置开始，直到写入全部完成。
- 它尝试将 `buf` 中的数据写入底层的 TCP 连接。
- 返回值是**实际写入的字节数**，可能小于 `buf.len()`。
- TCP 是**流式协议**，没有消息边界：
-  发送缓冲区有限，当缓冲区满时，`write` 不能写入所有数据，只写一部分。
- 操作系统底层的缓冲机制和网络拥塞控制等都会影响写入的数量。
- 因此，调用 `write` 后，有可能只写了部分数据，需要多次调用才能发送完全部数据。
最后，我们执行刷新( flush )操作，确保写入操作已执行完毕
- `TcpStream::flush()` 在标准库中**就是个空操作（no-op）**，不会触发任何系统调用，不会强制发送缓冲数据。

好了，现在我们有了一个能工作的 Web 服务！
```bash
$ curl localhost:3000
```

- **Windows 上，VSCode 打开终端默认启动 PowerShell**，因为它是 Windows 10/11 自带的现代命令行环境，功能丰富，支持脚本、自动补全等。
- 所以你在 VSCode 终端里敲 `curl`，其实是调用 PowerShell 的 `Invoke-WebRequest` 别名，而不是系统的原生 `curl.exe`。
	- `curl` 是 `Invoke-WebRequest` 的别名（alias）。
	- `Invoke-WebRequest` 是一个功能强大的 HTTP 客户端，设计时就考虑了脚本和自动化的需求。
	- 它不仅会把网页内容作为字符串返回，还会把**响应头、Cookies、状态码、重定向信息、HTML DOM 结构、HTTP 版本等信息都以对象属性的形式封装返回**。
	- `curl.exe` 只输出响应体文本（或者二进制数据），如果用 `-v` 选项才会显示更详细的头部信息。
	- PowerShell 的 `Invoke-WebRequest` 默认就是返回丰富的结构化数据，方便脚本调用。
- HTTP 明确规定：
	- 每个响应头必须以 `\r\n` 结尾；
	- 所有头部q之后，需要 **额外的 `\r\n`** 表示头部结束；
	- 所以你最后一行是 `\r\n\r\n`，用于标识正文开始没问题。
	如果只用了 `\n`，像 `PowerShell curl`（即 `Invoke-WebRequest`）这类工具就会报错。
	- 但是原生不会报错
		- curl 是一个成熟的跨平台工具，经过多年发展，设计上更注重兼容性和容错能力。
		- 它在处理服务器响应时，遇到协议上细微不规范（比如换行符格式不完全正确）时，通常不会报错，而是尽量“修正”或忽略这些小瑕疵，保证请求能正常完成。
		- 所以即使你的服务器响应头部换行使用了不规范的换行方式（比如只用 CR 或 LF，或者顺序不正确），curl.exe 依然能正常处理。
vscode终端调用：curl http://localhost:3000
结果：
```rust
StatusCode        : 200
StatusDescription : OK
Content           : {72, 101, 108, 108...}
RawContent        : HTTP/1.1 200 OK
                    Connection: close
                    Content-Length: 12

                    Hello world!
Headers           : {[Connection, close], [Content-Length,
                     12]}
RawContentLength  : 12

```
