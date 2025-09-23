你遇到的错误：

```perl
Liquid Exception: SSL_connect returned=1 errno=107 peeraddr=(null) state=error: unexpected eof while reading
```

这是 Ruby/Jekyll 在处理 HTTPS 请求或生成静态页面时遇到 **SSL/TLS 连接异常** 导致的。它通常出现在：

-   插件尝试抓取 HTTPS 外部资源（RSS/网页）时。
-   Jekyll 处理某篇 Markdown 文件中包含远程资源时（如图片或外部链接）。
-   OpenSSL 或系统 CA 证书有问题。
---

### 可能原因

1.  **外部 RSS 或 URL 的 HTTPS 证书问题**
    
    -   证书过期、域名不匹配或 TLS 版本不兼容。
        
2.  **Docker / 环境中的 OpenSSL**
    
    -   Docker 镜像可能缺少 CA 根证书。
        
    -   Ruby 使用的 OpenSSL 版本过旧。
        
3.  **网络中断或代理**
    
    -   SSL 握手未完成，导致 EOF。
        

---

### 解决方法

#### 1️⃣ 临时跳过 SSL 验证（不推荐生产）

在抓取 RSS 或网页的地方，添加 `verify: false`：

```ruby
xml = HTTParty.get(src['rss_url'], verify: false).body
```

或：

```ruby
html = HTTParty.get(url, verify: false).body
```

> 注意：这样会忽略 SSL 校验，存在安全风险。

---

#### 2️⃣ 安装 CA 证书（推荐）

-   在 Docker 容器里安装 `ca-certificates`：
    

```bash
apt-get update && apt-get install -y ca-certificates
update-ca-certificates
```

-   确保 Ruby 使用系统证书：
    

```bash
gem install openssl
```

---

#### 3️⃣ 使用 HTTP 版本的 RSS / URL

-   如果源站支持 HTTP，可以临时改为：
    

```yaml
rss_url: http://medium.com/@al-folio/feed
```

> 仅做测试，不建议长期使用 HTTP。

---

#### 4️⃣ 检查特定 Markdown 文件

错误信息中提到：

```swift
/srv/jekyll/_posts/2020-09-28-twitter.md
```

-   检查该文件中是否有远程资源（图片、iframe、外部链接）。
    
-   确认这些 URL 是否可以访问，或者临时注释掉相关内容。
    

---

### 5️⃣ 最安全的做法

-   确保 Docker 容器能访问外网 HTTPS。
    
-   安装 CA 根证书。
    
-   更新 Ruby + OpenSSL 版本。
    
-   只在可信环境下抓取 RSS/网页。
    
