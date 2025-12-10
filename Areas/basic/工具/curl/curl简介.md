### 1. `curl` 简介

`curl` 是一个命令行工具，用于在命令行或脚本中进行网络请求，支持 HTTP、HTTPS、FTP、SMTP 等多种协议。它可以用来下载文件、发送 API 请求、测试接口等。`curl` 的核心优势在于灵活、可组合性强、适合脚本自动化。

---

### 2. 基本用法

#### 2.1 下载网页或文件

```bash
curl https://example.com
```

- 默认输出内容到终端。
- 使用 `-o` 指定保存文件名：

```bash
curl -o index.html https://example.com
```

#### 2.2 显示 HTTP 头部

```bash
curl -I https://example.com
```

- `-I` 或 `--head` 仅获取响应头，不下载内容。
    

#### 2.3 保存并显示进度条

```bash
curl -O https://example.com/file.zip
```

- `-O` 自动使用远程文件名保存。
    
- `-#` 显示进度条：
    

```bash
curl -O -# https://example.com/file.zip
```

---

### 3. 发送 GET 请求

默认就是 GET 请求，可以带参数：

```bash
curl "https://api.example.com/data?user=alice&age=20"
```

- 注意 URL 参数需用引号包裹，防止特殊字符被 shell 解析。
    

---

### 4. 发送 POST 请求

#### 4.1 表单提交

```bash
curl -X POST -d "username=alice&password=123456" https://api.example.com/login
```

- `-d` 或 `--data` 指定表单数据。
    
- `-X POST` 明确使用 POST 方法（有 `-d` 时可以省略）。
    

#### 4.2 JSON 数据

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"username":"alice","password":"123456"}' \
     https://api.example.com/login
```

- `-H` 指定请求头。
    

---

### 5. 常用选项

|选项|说明|
|---|---|
|`-H`|添加 HTTP 头|
|`-u`|用户名密码认证，如 `-u user:pass`|
|`-L`|跟随重定向|
|`-v`|显示详细请求和响应信息|
|`-k`|忽略 HTTPS 证书验证（不推荐生产使用）|
|`-I`|只请求头部|
|`-o`|保存输出到文件|
|`-O`|使用远程文件名保存|

---

### 6. 高级用法

#### 6.1 上传文件

```bash
curl -F "file=@localfile.txt" https://api.example.com/upload
```

- `-F` 用于 multipart/form-data 上传。
    

#### 6.2 发送自定义请求

```bash
curl -X DELETE https://api.example.com/item/123
```

#### 6.3 使用 Cookie

```bash
curl -b "name=value" https://example.com
curl -c cookies.txt https://example.com  # 保存 cookie
curl -b cookies.txt https://example.com  # 使用 cookie
```

#### 6.4 设置超时

```bash
curl --connect-timeout 5 --max-time 10 https://example.com
```

- `--connect-timeout` 连接超时。
    
- `--max-time` 总请求超时。
    

#### 6.5 并发请求（结合 xargs 或 GNU parallel）

```bash
cat urls.txt | xargs -n 1 -P 5 curl -O
```

- `-P 5` 表示最多 5 个并发下载。
    

---

### 7. 学习方法与练习

1. 从基础命令开始：
    
    - GET、POST、文件下载。
        
2. 结合 API 接口练习：
    
    - RESTful API 调用。
        
    - 使用 JSON 数据和表单数据。
        
3. 学习 HTTP 协议：
    
    - 状态码、请求头、响应头、Cookie、重定向。
        
4. 练习脚本自动化：
    
    - 批量下载。
        
    - 自动化接口测试。
        
5. 使用 `-v` 或 `--trace` 调试请求：
    
    - 观察请求过程，理解网络交互。
        

---

**总结**  
`curl` 是网络请求的万能工具，理解它不仅有助于抓取网页和文件，也能帮助测试 API、调试网络问题。重点是熟练掌握 GET/POST/HTTP 头/JSON 数据/文件上传这些基础操作，进阶可以学习脚本化批量操作与调试技巧。通过不断练习真实接口和文件下载，可以快速提高实战能力。

---

如果你需要，我可以帮你整理一份 **30 条 `curl` 常用命令速查表**，方便随时查用。你希望我整理吗？