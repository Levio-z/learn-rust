---
tags:
  - permanent
---
## 1. 核心观点  

#### 定义

在 Git 的 **Smart HTTP 协议** 中，`/info/refs?service=git-upload-pack` 是一个特定的 URL 请求，用于**获取服务器端的引用（refs）列表及 capabilities 信息**。

- `git-upload-pack` 是服务器端用于处理客户端 `git fetch` 或 `git clone` 请求的服务命令。
    
- 客户端通过这个请求了解服务器上有哪些分支、标签以及支持的协议能力，从而决定后续增量拉取哪些对象。
    

核心功能是**确认服务器拥有而客户端没有的数据**，并为后续拉取做准备。

## 2. 背景/出处  
- 来源：https://git-scm.com/docs/gitprotocol-pack
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
#### 3.1 核心内容和实际解析
核心内容和实际解析
```
 advertised-refs  =  *1("version 1")
		      (no-refs / list-of-refs)
		      *shallow
		      flush-pkt

  no-refs          =  PKT-LINE(zero-id SP "capabilities^{}"
		      NUL capability-list)

  list-of-refs     =  first-ref *other-ref
  first-ref        =  PKT-LINE(obj-id SP refname
		      NUL capability-list)

  other-ref        =  PKT-LINE(other-tip / other-peeled)
  other-tip        =  obj-id SP refname
  other-peeled     =  obj-id SP refname "^{}"

  shallow          =  PKT-LINE("shallow" SP obj-id)

  capability-list  =  capability *(SP capability)
  capability       =  1*(LC_ALPHA / DIGIT / "-" / "_")
  LC_ALPHA         =  %x61-7A
```

实际解析

```
001e# service=git-upload-pack
0000015b9e4257ec52490078b918ab43831520d495e2a75e HEAD^@multi_ack thin-pack side-band side-band-64k ofs-delta shallow deepen-since deepen-not deepen-relative no-progress include-tag multi_ack_detailed allow-tip-sha1-in-want allow-reachable-sha1-in-want no-done symref=HEAD:refs/heads/master filter object-format=sha1 agent=git/github-60d715541676-Linux
003f9e4257ec52490078b918ab43831520d495e2a75e refs/heads/master
0000
```

```
001e# service=git-upload-pack
```
- **pkt-len** = `0x001e` = 30（十进制）
    - 说明整条 pkt-line 有 30 个字节，包括前 4 个长度字节
- **pkt-payload** 字节数 = pkt-len - 4 = 30 - 4 = 26
	- （”# service=git-upload-pack\n“）加上换行符
>这块和官网的协议不同
```
  flush-pkt：0000
```
>这块和官网的协议不同

```
长度 015b：347
载荷 PKT-LINE(obj-id SP refname NUL capability-list)：9e4257ec52490078b918ab43831520d495e2a75e HEAD^@multi_ack thin-pack side-band side-band-64k ofs-delta shallow deepen-since deepen-not deepen-relative no-progress include-tag multi_ack_detailed allow-tip-sha1-in-want allow-reachable-sha1-in-want no-done symref=HEAD:refs/heads/master filter object-format=sha1 agent=git/github-60d715541676-Linux\n
```

```
长度：003f 59
载荷 other-ref        =  PKT-LINE(other-tip / other-peeled)）：9e4257ec52490078b918ab43831520d495e2a75e refs/heads/master\n
```



#### 3.2原理

1. **HTTP GET 请求**
    

客户端访问：

```
GET /<repository>/info/refs?service=git-upload-pack
```

2. **服务器响应**
    

- 返回一个 `pkt-line` 格式的流，其中包含：
    
    - 所有引用名和对应 commit 哈希（refs）
        
    - 支持的 capabilities，例如 `multi-ack`, `thin-pack`, `ofs-delta` 等
        

示例（简化）：

```
001e# service=git-upload-pack
0000
003f9fceb02... refs/heads/main
0040a1b2c3d4... refs/tags/v1.0
```

- 第一行是协议标识 `# service=git-upload-pack`
    
- 后续是各引用及其对应对象哈希
    
- `0000` 表示 flush-pkt，结束信息
    

3. **客户端处理**
    

- 解析返回的 refs，确定本地缺失的 commit 或对象。
    
- 准备发送 `git-upload-pack` 请求，服务器仅返回缺失对象，实现**增量同步**。
    

---



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- 【ref】[智能协议测试](智能协议测试.md)
	- [Git-URL Format  URL 格式](Git-URL%20Format%20%20URL%20格式.md)
	- [Git-gitprotocol-pack - pkt-line 格式](Git-gitprotocol-pack%20-%20pkt-line%20格式.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
