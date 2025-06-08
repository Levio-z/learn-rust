```rust
let response = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Length: 12\n",
    "Connection: close\r\n\r\n",
    "Hello world!"
);
```
### 逐段解析

|片段|作用|
|---|---|
|`"HTTP/1.1 200 OK\r\n"`|响应行，表明 HTTP 版本和状态码|
|`"Content-Length: 12\n"`|响应头，声明正文长度是12字节（注意用的是 `\n`）|
|`"Connection: close\r\n\r\n"`|响应头，告诉客户端关闭连接；后面两个换行表示头部结束|
|`"Hello world!"`|HTTP 响应正文，内容是 `Hello world!`|
**`Connection: close\r\n`**  
这是 HTTP 响应头的一部分，告诉客户端：
- 服务器处理完当前请求后，会关闭 TCP 连接，不会保持长连接（Keep-Alive）。
- 这对于简单的 HTTP/1.0 或不支持长连接的客户端非常常见。