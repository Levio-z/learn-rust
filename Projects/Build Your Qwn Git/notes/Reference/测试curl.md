curl -v --http1.0 "$GIT_URL/info/refs"

### `curl -v --http1.0 "$GIT_URL/info/refs"` 的定义

这条命令用于向 Git HTTP 服务器发送 **GET /info/refs** 请求，并强制使用 **HTTP/1.0** 协议，同时输出详细调试信息（verbose）。这是 Git Smart HTTP/Dumb HTTP 协议调试中最常用的方式之一。

---

### ### 命令的组成与作用

#### **1. `curl`**

命令行下发起 HTTP/HTTPS 请求的工具，是 Git HTTP 协议调试的基础。

#### **2. `-v`（verbose）**

打印完整请求与响应过程，包括：

- 请求头
    
- 响应头
    
- 状态码
    
- TCP 连接过程
    
- 数据包内容（文本形式）
    

便于分析 Git 服务器是否正确暴露 `info/refs`。

#### **3. `--http1.0`**

强制使用 **HTTP/1.0** 发起请求，默认 curl 是 HTTP/1.1。

含义与影响：

- 不支持 keep-alive（除非额外加入头）
    
- 无 chunked 传输
    
- Git 服务器若在严格模式下，可能返回不同内容（尤其 Smart HTTP）
    
- 更接近某些嵌入式 Git 服务器的兼容模式
    

这是你在调试 Git 低版本或定制 Git 服务端时常需要的参数。

#### **4. `"$GIT_URL/info/refs"`**

访问 Git 仓库的 `info/refs` 静态索引文件。其作用：

- 列出远端所有 refs（分支、tag）
    
- Smart HTTP 模式下用于广告 capability
    
- Dumb HTTP 模式下用于告诉客户端可用对象
    

等价于浏览器打开：

```
https://example.com/myrepo.git/info/refs
```

---

### ### 该命令在 Git 协议中的地位

Git clone 的第一步就是访问 info/refs：

1. GET `/info/refs?service=git-upload-pack`（Smart HTTP）
    
2. GET `/info/refs`（Dumb HTTP）
    

而你这条命令模拟的是 Dumb HTTP 风格的访问流程。

这非常适合用于：

- 检查仓库是否能被浏览器访问
    
- 测试 Web 服务器配置是否阻挡 Git 文件
    
- 调试 Smart/Dumb HTTP 模式切换是否正确
    
- 查看 Git 服务端是否返回正确的 refs 索引
    
