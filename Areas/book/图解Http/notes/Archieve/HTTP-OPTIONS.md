---
tags:
  - note
---

**OPTIONS 方法** 的核心用途就是：
### 核心作用：查询支持的方法 (Allowed Methods)
- **目的：** 允许客户端请求服务器，以了解针对特定的请求 URI，服务器支持哪些 HTTP 方法（如 GET, POST, PUT, DELETE 等）。
- **实现方式：** 服务器在响应 `OPTIONS` 请求时，会在响应头部中包含一个特殊的字段：**`Allow`**。
#### 示例
客户端发送请求：
HTTP
```
OPTIONS /users HTTP/1.1
Host: api.example.com
```
服务器返回响应：
HTTP
```
HTTP/1.1 200 OK
Allow: GET, POST, HEAD, OPTIONS
Content-Length: 0
```
在这个例子中，`Allow` 头部告诉客户端，它可以通过 `GET`、`POST`、`HEAD` 和 `OPTIONS` 方法来操作 `/users` 这个资源。
### 其他重要用途：CORS 预检请求
在现代 Web 开发中，`OPTIONS` 方法还有一个极其重要的用途，那就是作为 **CORS (Cross-Origin Resource Sharing，跨域资源共享)** 机制中的**预检请求 (Preflight Request)**。
当浏览器尝试发送某些复杂的跨域请求（如带有自定义头、或使用 PUT、DELETE 等方法）时，浏览器会先自动发送一个 `OPTIONS` 预检请求：
1. 预检请求询问服务器，是否允许来自该源（Origin）的请求，以及是否支持特定的 HTTP 方法和头部。
2. 服务器通过响应头（如 `Access-Control-Allow-Origin` 和 `Access-Control-Allow-Methods`）回答。
3. 如果预检通过，浏览器才会发送实际的跨域请求。
因此，**OPTIONS 方法** 在**资源能力查询**和**跨域安全控制**方面都扮演着关键角色。