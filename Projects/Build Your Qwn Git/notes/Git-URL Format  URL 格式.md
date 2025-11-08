---
tags:
  - note
---
## 1. 核心观点  

```
http://<host>:<port>/<path>?<searchpart>
```
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 标准语法
HTTP 访问的 Git 存储库的 URL 使用 RFC 1738 中记录的标准 HTTP URL 语法
```
http://<host>:<port>/<path>?<searchpart>
```

占位符 `$GIT_URL` 将代表最终用户输入的 http:// 存储库 URL。

|             |                                                                   |
| ----------- | ----------------------------------------------------------------- |
| 协议          | 客户端请求的路径组件示例                                                      |
| **哑 HTTP**  | `**/HEAD**`, `**/objects/info/packs**`, `**/objects/<sha1>/...**` |
| **智能 HTTP** | `**/info/refs?service=git-upload-pack**` (拉取请求)                   |
|             | `**/git-upload-pack**` (实际数据传输)                                   |
|             | `**/git-receive-pack**` (推送请求)                                    |
### 案例
An example of a dumb client requesting for a loose object:  
请求松散对象的哑客户端示例：

```
$GIT_URL:     http://example.com:8080/git/repo.git
URL request:  http://example.com:8080/git/repo.git/objects/d0/49f6c27a2244e12041955e262a404c7faba355

```
An example of a smart request to a catch-all gateway:  
对捕获网关的智能请求示例：

```
$GIT_URL:     http://example.com/daemon.cgi?svc=git&q=
URL request:  http://example.com/daemon.cgi?svc=git&q=/info/refs&service=git-receive-pack

```
**当用户执行 `git push` 到这个 URL 时，智能客户端会构建一个 HTTP 请求，告诉服务器：**

1. **目标程序：** 请运行位于 `daemon.cgi` 的网关程序。
    
2. **目标操作：** 我想使用 **`git-receive-pack` 服务**（即我要推送数据）。
    
3. **获取信息：** 在推送之前，请先给我 `/info/refs` 的信息。
An example of a request to a submodule:  
对子模块的请求示例：

```
$GIT_URL:     http://example.com/git/repo.git/path/submodule.git
URL request:  http://example.com/git/repo.git/path/submodule.git/info/refs
```
### 服务器要求

因为 Git 客户端会通过向基础 `$GIT\\_URL` **动态追加路径**来请求仓库的不同部分、元数据或服务，所以 Web 服务器（或其配置的 CGI/应用）必须被设置为：

**接管**任何以 `$GIT\\_URL` 开头的请求，并根据追加的路径组件来确定是返回一个静态文件（哑协议）还是启动一个 Git 服务程序（智能协议）。这意味着服务器不能只配置一个静态的 URL 路径来提供 Git 仓库。它需要一个**程序或脚本**来接管对该基础 URL 路径下**所有子路径**的请求。

这强调了 Git over HTTP 访问依赖于服务器侧的**路由和应用程序逻辑**，而不仅仅是简单的文件服务。
#### 剥离尾随
- **剥离尾随 `/`**: 客户端在处理用户提供的 `$GIT_URL` 字符串时，**必须**去除末尾的斜杠（如果存在）。
- **避免空路径令牌**: 这样做的目的是**防止**在发送到服务器的任何 URL 中出现空路径令牌（通常是两个连续的斜杠 `//` 导致）。
- **兼容性要求**: 兼容的客户端必须能够将 `$GIT_URL/info/refs` 正确地展开为 `foo/info/refs`，而不是带有双斜杠的 `foo//info/refs`（假设 `foo` 是剥离 `/` 后的 `$GIT_URL`）。



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
