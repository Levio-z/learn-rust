 ### Git `info/refs` Wire Protocol 数据解析

---

 ### 一、整体结构定义

你看到的这段内容是 **Git Smart HTTP 协议**（upload-pack 服务）返回的 **pkt-line 格式（Packet Line）** 数据。它是 Git 服务端告诉客户端当前仓库有哪些引用（refs）、HEAD 指向哪里、支持哪些能力（capabilities）。

Git 在 Smart HTTP 下通过 pkt-line 编码，每一行前 4 字节为十六进制长度字段。

格式如下：

```
LLLL<内容>
```

- **LLLL**：4 字节十六进制，表示整个包的长度（含自身）
    
- **内容**：后续数据
    
- `0000` 代表 **flush 包**（结束）
    

---

### ### ### 二、逐行解释源码

#### 1. `001e# service=git-upload-pack`

- `001e` = 0x1e = 30 字节长度
- 内容是：
    ```
    # service=git-upload-pack
    ```
    
- 意义：服务端告诉客户端当前提供的 Git 服务类型是 `upload-pack`，即 “拉取” 服务。

---

#### 2. `0000`
- flush 包，表示 service 宣告区结束。
---

#### 3. `015b9e4257ec52490078b918ab43831520d495e2a75e HEAD^@...`

解析关键步骤：

- `015b` → 0x015b = **347 字节长度**
    
- 从第五个字节起是 actual payload：
    

内容结构如下：

```
<40字节commit-id> <refname>\0<capabilities>
```

拆解：

1. commit-id：
    
    ```
    9e4257ec52490078b918ab43831520d495e2a75e
    ```
    
    这是 HEAD 所指向的 commit。
    
2. 引用名：
    
    ```
    HEAD
    ```
    
3. `^@`（即 `\x00`）：分隔符，后面是能力（capabilities）
    
4. capabilities（核心内容）：
    
    ```
    multi_ack thin-pack side-band side-band-64k ofs-delta shallow deepen-since deepen-not deepen-relative no-progress include-tag multi_ack_detailed allow-tip-sha1-in-want allow-reachable-sha1-in-want no-done symref=HEAD:refs/heads/master filter object-format=sha1 agent=git/github-60d715541676-Linux
    ```
    

每个 capability 都定义 upload-pack 行为，例如：

- **thin-pack**：生成更小的 packfile
    
- **side-band / side-band-64k**：允许 Git 在一个流中同时发送数据 / 进度
    
- **shallow**：支持浅克隆
    
- **symref=HEAD:refs/heads/master**：声明 HEAD 实际是一个符号引用，指向 `master`
    

---

#### 4. `003f9e4257ec52490078b918ab43831520d495e2a75e refs/heads/master`

- `003f` → 长度 63 字节
    
- 内容格式同前：
    

```
<commit-id> <refname>
```

解析：

- commit-id 同上
    
- refname = `refs/heads/master`
    

说明这个仓库只有一个分支：master。
