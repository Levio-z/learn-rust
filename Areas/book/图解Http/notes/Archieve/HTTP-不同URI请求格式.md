---
tags:
  - note
---

你所说的**非常正确**。在客户端（例如浏览器）通过 HTTP 协议请求访问资源时，URI 是请求报文中**核心且必不可少**的一部分。

在 HTTP 请求报文中，**请求 URI (Request-URI)** 的指定方式确实有很多种，这主要取决于 HTTP 协议的版本、请求的方法（Method）以及请求的上下文（例如是否使用代理）。

以下是指定请求 URI 的几种主要方式：

### 1. 原始格式 (Original Form - 最常见)

这是客户端向源服务器（Origin Server）发送请求时最常用的形式，URI 仅包含**资源的路径和查询部分**。
- **用于：** 客户端直接向服务器请求（无代理或使用透明代理）。
- **示例 (HTTP/1.1):**
    HTTP
    ```
    GET /path/to/resource?query=1 HTTP/1.1
    Host: www.example.com
    ```
    请求 URI 是：`/path/to/resource?query=1`
    

### 2. 完整绝对 URI 格式 (Absolute URI Form)
在这种情况下，请求 URI 是一个**完整的绝对 URI**，包括协议名、主机和端口。
- **用于：**
    - **HTTP 代理服务器 (Proxy):** 当客户端向代理服务器发送请求时，必须使用完整的 URI，这样代理服务器才知道要连接哪个目标服务器。
    - **某些旧版或非标准的请求。**
- **示例 (HTTP/1.1 通过代理):**
    HTTP
    ```
    GET http://www.example.com/path/to/resource?query=1 HTTP/1.1
    Host: www.example.com
    ```
    请求 URI 是：`http://www.example.com/path/to/resource?query=1`
### 3. Authority 格式

这种格式仅使用 URI 中的 **Authority（主机和端口）** 部分。
- **用于：** **CONNECT 方法**，通常用于建立到目标服务器的安全隧道（如 HTTPS）穿越代理。
- **示例 (通过代理建立隧道):**
    HTTP
    ```
    CONNECT www.example.com:443 HTTP/1.1
    Host: www.example.com:443
    ```
    
    请求 URI 是：`www.example.com:443`
    

### 4. 星号格式 (Asterisk Form)
在这种格式中，请求 URI 被一个单独的星号（`*`）代替。
- **用于：** **OPTIONS 方法**，当请求是针对服务器**整体**而非特定资源时。
- **示例:**
    HTTP
    ```
    OPTIONS * HTTP/1.1
    Host: www.example.com
    ```
    
    请求 URI 是：`*`
    

这些不同的格式体现了 HTTP 协议在不同场景和功能（直接访问、代理、隧道、服务器选项）下对 URI 的灵活运用。