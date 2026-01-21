### 1. 协议簇概述与分层模型 (Overview & Layer Model)

- **1.1 TCP/IP 历史与起源：** 从 ARPANET 到互联网。
    
- **1.2 TCP/IP 结构模型：**
    
    - **四层模型（经典）：** 应用层、传输层、网络层、网络接口层。
        
    - 与 OSI 七层模型的对比和映射关系。
        
- **1.3 数据封装与解封装：** PDU（协议数据单元）在各层间的变化（数据、段、包、帧）。
    

### 2. 网络接口层 / 链路层 (Link Layer)

- **2.1 硬件寻址：** MAC 地址（物理地址）的格式与作用。
    
- **2.2 以太网 (Ethernet)：** 规范、拓扑结构。
    
- **2.3 ARP (Address Resolution Protocol)：** 地址解析协议。
    
    - 目的：将 IP 地址映射到 MAC 地址。
        
    - ARP 请求与响应过程。
        
- **2.4 RARP (Reverse ARP)：** 反向地址解析协议（了解）。
    

### 3. 网络层 / 互联网层 (Internet Layer)

- **3.1 核心协议：IP (Internet Protocol)：**
    
    - **主要功能：** 寻址与路由（Routing）。
        
    - **特性：** 无连接、不可靠交付（尽力而为）。
        
- **3.2 IP 地址系统：**
    
    - **IPv4：** 地址结构、子网划分 (Subnetting)、CIDR (无类别域间路由)。
        
    - **IPv6：** 巨大的地址空间、新的头部结构。
        
- **3.3 IP 数据报结构：** 头部字段、分片与重组。
    
- **3.4 ICMP (Internet Control Message Protocol)：** 互联网控制消息协议。
    
    - 功能：差错报告与查询信息（如 Ping, Traceroute 的原理）。
        
- **3.5 路由协议（Routing Protocols）：**
    
    - **内部网关协议 (IGP)：** RIP, OSPF。
        
    - **外部网关协议 (EGP)：** BGP。
        

### 4. 传输层 (Transport Layer)

- **4.1 端口号 (Port Numbers)：** 寻址到特定应用进程（知名端口、注册端口、动态端口）。
    
- **4.2 TCP (Transmission Control Protocol)：**
    
    - **核心功能：** 可靠性、连接管理、流量控制、拥塞控制。（详细内容请参考您上一份笔记）。
        
    - **关键机制：** 三次握手、四次挥手、序列号/确认号、滑动窗口。
        
- **4.3 UDP (User Datagram Protocol)：**
    
    - **核心功能：** 无连接、不可靠、低延迟。
        
    - **适用场景：** DNS、视频/音频流、网络游戏。
        
- **4.4 TCP 与 UDP 的比较：** 特点、开销与适用场景。
    

### 5. 应用层 (Application Layer)

- **5.1 域名系统 (DNS)：**
    
    - 功能：将域名解析为 IP 地址。
        
    - DNS 层次结构：根服务器、顶级域、权威域名服务器。
        
    - DNS 记录类型。
        
- **5.2 Web 相关协议：**
    
    - **HTTP (Hypertext Transfer Protocol)：** 无状态、请求/响应模型。
        
    - **HTTPS：** HTTP + TLS/SSL 加密。
        
- **5.3 电子邮件协议：**
    
    - **SMTP (Simple Mail Transfer Protocol)：** 邮件发送。
        
    - **POP3/IMAP：** 邮件接收与管理。
        
- **5.4 文件传输与远程控制：**
    
    - **FTP (File Transfer Protocol)：** 文件传输（控制连接与数据连接）。
        
    - **SSH (Secure Shell)：** 安全的远程登录与命令执行。
        
- **5.5 动态主机配置：**
    
    - **DHCP (Dynamic Host Configuration Protocol)：** 自动分配 IP 地址。
        

### 6. 安全机制 (Security & Attacks)

- **6.1 TLS/SSL (Transport Layer Security)：**
    
    - 功能：在传输层提供数据加密、认证和完整性保护。
        
    - 握手过程（公钥、私钥、会话密钥）。
        
- **6.2 防火墙 (Firewalls)：** 各种类型和过滤规则。
    
- **6.3 常见网络攻击：**
    
    - 拒绝服务攻击 (DoS/DDoS)。
        
    - 中间人攻击 (Man-in-the-Middle)。
        
    - 端口扫描。
        

### 7. 实践与工具 (Practical Tools & Implementation)

- **7.1 网络工具：** `ping`、`traceroute`/`tracert`、`netstat`。
    
- **7.2 数据包分析：** Wireshark 的使用和原理（捕获和分析协议数据）。
    
- **7.3 Socket 编程基础：** 了解应用如何通过 Socket API 调用 TCP/UDP 服务。