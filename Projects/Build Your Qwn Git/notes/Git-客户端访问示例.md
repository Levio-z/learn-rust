
请求松散对象的哑客户端示例：

$GIT_URL:     http://example.com:8080/git/repo.git
URL request:  http://example.com:8080/git/repo.git/objects/d0/49f6c27a2244e12041955e262a404c7faba355


对捕获网关的智能请求示例：

$GIT_URL:     http://example.com/daemon.cgi?svc=git&q=
URL request:  http://example.com/daemon.cgi?svc=git&q=/info/refs&service=git-receive-pack

对子模块的请求示例：

$GIT_URL:     http://example.com/git/repo.git/path/submodule.git
URL request:  http://example.com/git/repo.git/path/submodule.git/info/refs


客户端必须从用户提供的内容中剥离尾随 `/`（如果存在） `$GIT_URL` 字符串，以防止出现空路径令牌 （） 在发送到服务器的任何 URL 中。 兼容客户端必须展开 `$GIT_URL/info/refs` 作为 `foo/info/refs` 而不是 `foo//info/refs`。

[Git-两种传输协议](Git-两种传输协议.md)