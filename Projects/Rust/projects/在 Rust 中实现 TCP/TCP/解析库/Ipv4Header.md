```rust
header,
Ipv4Header {
    total_len: (100 + Ipv4Header::MIN_LEN) as u16,
    time_to_live: 4,
    protocol: IpNumber::UDP,
    source: [1, 2, 3, 4],
    destination: [5, 6, 7, 8],
    ..Default::default()
}

```
#### ✅ `total_len: (100 + Ipv4Header::MIN_LEN) as u16`

- **定义**：IPv4 报文总长度字段（单位：字节），包含：IP首部 + IP负载（如UDP/TCP等）。
    
- **这里含义**：假设 UDP Payload 是 100 字节，IPv4 头部最小长度通常为 20 字节（`Ipv4Header::MIN_LEN`），所以总长为 `120`。
    
- **原理**：IP 层需要告诉网络层整个包多大，以便路由器判断是否需要分片。
    
- **注意**：最大值为 `65535`，因为该字段是 `u16`。
    
- **扩展**：如果启用 IP 选项字段，`MIN_LEN` 会变长。
    

#### ✅ `time_to_live: 4`

- **定义**：TTL（生存时间），每经过一个路由器减一，减到 0 就丢弃。
    
- **这里含义**：最多允许跨越 4 个路由器。
    
- **原理**：防止 IP 分组在环路中无限转发。
    
- **使用场景**：调试 traceroute 工具或构造环路测试。
    
- **扩展**：`traceroute` 正是通过逐步递增 TTL 得到每跳路由器的响应。
    

#### ✅ `protocol: IpNumber::UDP`

- **定义**：指定下层使用的协议类型（值取自 IANA 分配的 IP 协议号表）。
    
- **这里含义**：封装的是 UDP 报文（值为 `17`）。
    
- **其他常见值**：
    
    - TCP: `6`
        
    - ICMP: `1`
        
- **扩展**：如果你在实现 NAT 或防火墙，需要根据 protocol 区分处理路径。
    

#### ✅ `source: [1, 2, 3, 4]` 和 `destination: [5, 6, 7, 8]`

- **定义**：IPv4 源地址与目的地址（4 字节数组）。
    
- **作用**：确定包从哪里来、送到哪里去。
    
- **这里含义**：源IP是 `1.2.3.4`，目的IP是 `5.6.7.8`。
    
- **扩展**：在构造原始 socket 测试包、欺骗源地址时会改这个字段。
    

#### ✅ `..Default::default()`

- **定义**：其余字段使用 `Ipv4Header` 的默认值填充。
    
- **作用**：减少冗余代码，避免显式填写如 `identification`、`flags`、`header_checksum` 等字段。
    
- **原理**：Rust 的 `Default` trait 会初始化字段为 0 或默认安全值。
    
- **扩展**：可通过实现 `Default` 自定义初始化行为。
    

---

### 🚀 使用场景

- 编写网络协议测试工具（如 packet crafter、fuzzer）
    
- 进行网络安全演练（例如伪造 UDP flood）
    
- 自定义网络栈协议栈（如在操作系统课程中构建简化 IP 层）
    
- 教学演示，手动构造报文理解协议字段作用