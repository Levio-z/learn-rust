### 前置知识
- [协议作为规范的意思](../../../../book/图解Http/notes/Archieve/协议作为规范的意思.md)
- [HTTP-WWW基础理念](../../../../book/图解Http/notes/Archieve/HTTP-WWW基础理念.md)
### 初衷
- 参考：[万维网（WWW）时的基本理念和核心机制](../../../../book/图解Http/notes/Reference/万维网（WWW）时的基本理念和核心机制.md)
### 1️⃣ HTTP 协议基础
- [HTTP-基本概念](../../../../book/图解Http/notes/Archieve/HTTP-基本概念.md)
- **HTTP 请求结构**
    - Request-Line: 方法、路径、协议版本
    - Header: 常用字段（Host、User-Agent、Content-Type、Authorization、Cookie）
    - Body: GET/POST 数据格式、JSON、Form

- **HTTP 响应结构**
- Status-Line: 状态码、原因短语
- Header: Content-Type、Content-Length、Set-Cookie
- Body: JSON、HTML、二进制流

- **HTTP请求格式：**[HTTP-不同URI请求格式](../../../../book/图解Http/notes/Archieve/HTTP-不同URI请求格式.md)

- **常用方法**: [GET](../../../../book/图解Http/notes/Archieve/HTTP-GET.md)、[POST](../../../../book/图解Http/notes/Archieve/HTTP-POST.md)、[PUT](../../../../book/图解Http/notes/Archieve/HTTP-PUT.md)、[DELETE](../../../../book/图解Http/notes/Archieve/HTTP-DELETE.md)、PATCH、[OPTIONS](../../../../book/图解Http/notes/Archieve/HTTP-OPTIONS.md)、[CONNECT](../../../../book/图解Http/notes/Archieve/HTTP-CONNECT.md)、[HEAD](../../../../book/图解Http/notes/Archieve/HTTP-Head.md)

- **常用状态码**: 2xx 成功、3xx 重定向、4xx 客户端错误、5xx 服务端错误

- **Cookie / Session**: 存储机制、HTTPOnly、Secure、SameSite

- **URL 编码与 Query 参数**

### 2️⃣ HTTP 协议进阶

- **HTTP/1.1 特性**
    - Keep-Alive / Connection: 持久连接
    - Chunked Encoding 分块传输
    - Host 头与虚拟主机
        
- **HTTP/2 / HTTP/3**
    - Frame 类型、流多路复用
    - Header 压缩（HPACK / QPACK）
    - 连接升级与 TLS 基础
        
- **缓存机制**
    - Cache-Control、ETag、Last-Modified
        
- **请求重试与幂等性**
    - 幂等方法、重试策略
        
- **Content Negotiation**
    - Accept、Accept-Encoding、Content-Type

### 3️⃣ HTTP 安全与性能

- HTTPS / TLS 握手流程
- CORS 与跨域策略
- CSRF / XSS 基础防护
- HSTS、Strict-Transport-Security
- HTTP/2 Push 与性能优化
- 请求压缩 / 响应压缩（gzip、brotli）
- HTTP 流控与限流策略
### 4️⃣ Rust HTTP 实战

- **客户端**
    - `reqwest`：同步/异步请求，JSON 序列化/反序列化
    - `hyper`：低级 HTTP 客户端，流式处理

- **服务器**
    - `actix-web`：Actor 模型，高性能 Web 框架
    - `axum`：基于 Tower/Hyper，易组合中间件
    - `warp`：Filter 风格，路由与组合

- **中间件设计**
    - Logging、Auth、Rate Limiter、CORS

- **异步处理**
    - Tokio 异步任务调度
    - async fn / Future / Stream
    - HTTP 请求队列与并发控制

### 5️⃣ 高级实践

- HTTP Proxy / 网关设计
- 请求路由与负载均衡策略
- WebSocket 与 HTTP 协议升级
- Server-Sent Events (SSE)
- 流式响应与大文件传输
- 使用抓包工具分析请求（Wireshark、Chrome DevTools、Postman）

### 6️⃣ 面试/项目加分点

- 手写简单 HTTP 服务/客户端
- 使用 Rust 解析 HTTP 请求头/Body
- 使用 Hyper/Tokio 实现异步高并发 HTTP
- 理解 HTTP 状态码设计与 API 规范
- 实现中间件链（Logger / Auth / RateLimit）
- HTTP常见面试题

---
### 服务端与动态内容生成
- [HTTP-CGI](../../../../book/图解Http/notes/Archieve/HTTP-CGI.md)
### 💡 **学习方法论**：

1. **基础先行**：理解请求/响应结构、方法、状态码。
2. **实战演练**：用 Rust 实现简单 HTTP 服务和客户端。
3. **源码阅读**：分析 Hyper/Actix/Axum 的请求解析与路由机制。
4. **网络抓包**：观察真实请求与响应，理解 Header、Body、Chunked 等。
5. **进阶扩展**：HTTP/2、TLS、安全机制和性能优化。
### 参考资料
