_smoltcp_ 是一个独立的、事件驱动的 TCP/IP 堆栈，专为裸机实时系统而设计。其设计目标是简单性和健壮性。它的设计反目标包括复杂的编译时计算，例如宏或类型技巧，即使以性能下降为代价。
_smoltcp_ _根本不_需要堆分配，有[广泛的文档](https://docs.rs/smoltcp/) ，并且可以在稳定的 Rust 1.81 及更高版本上编译。
在环回模式下针对 Linux TCP 堆栈进行测试时，_smoltcp_ 可实现 [~Gbps 的吞吐量](https://github.com/smoltcp-rs/smoltcp#examplesbenchmarkrs) 。
### Features  特征
_SMOLTCP_ 缺少许多广泛部署的功能，通常是因为还没有人实现它们。为了正确设置预期，将列出已实现和省略的功能。
### Media layer  媒体层
There are 3 supported mediums.  
有 3 种受支持的媒体。

- Ethernet   以太网
    - Regular Ethernet II frames are supported.  
        支持常规以太网 II 帧。
    - Unicast, broadcast and multicast packets are supported.  
        支持单播、广播和组播数据包。
    - ARP packets (including gratuitous requests and replies) are supported.  
        支持 ARP 数据包（包括免费请求和回复）。
    - ARP requests are sent at a rate not exceeding one per second.  
        ARP 请求的发送速率不超过每秒 1 个。
    - Cached ARP entries expire after one minute.  
        缓存的 ARP 条目将在 1 分钟后过期。
    - 802.3 frames and 802.1Q are **not** supported.  
        **不支持** 802.3 帧和 802.1Q。
    - Jumbo frames are **not** supported.  
        **不支持**巨型帧。
- IP   知识产权
    - Unicast, broadcast and multicast packets are supported.  
        支持单播、广播和组播数据包。
- IEEE 802.15.4   IEEE 802.15.4 标准
    - Only support for data frames.  
        仅支持数据帧。
#### IPv4  IPv4 协议

[](https://github.com/smoltcp-rs/smoltcp#ipv4)

- IPv4 header checksum is generated and validated.  
    生成并验证 IPv4 标头校验和。
- IPv4 time-to-live value is configurable per socket, set to 64 by default.  
    IPv4 生存时间值可按套接字配置，默认设置为 64。
- IPv4 default gateway is supported.  
    支持 IPv4 默认网关。
- Routing outgoing IPv4 packets is supported, through a default gateway or a CIDR route table.  
    支持通过默认网关或 CIDR 路由表路由传出 IPv4 数据包。
- IPv4 fragmentation and reassembly is supported.  
    支持 IPv4 分段和重组。
- IPv4 options are **not** supported and are silently ignored.  
    IPv4 选项**不受**支持，并且将被静默忽略。

#### IPv6  IPv6 协议

[](https://github.com/smoltcp-rs/smoltcp#ipv6)

- IPv6 hop-limit value is configurable per socket, set to 64 by default.  
    IPv6 hop-limit 值可按套接字配置，默认设置为 64。
- Routing outgoing IPv6 packets is supported, through a default gateway or a CIDR route table.  
    支持通过默认网关或 CIDR 路由表路由传出 IPv6 数据包。
- IPv6 hop-by-hop header is supported.  
    支持 IPv6 逐跳报头。
- ICMPv6 parameter problem message is generated in response to an unrecognized IPv6 next header.  
    ICMPv6 参数问题消息是为了响应无法识别的 IPv6 next 报头而生成的。
- ICMPv6 parameter problem message is **not** generated in response to an unknown IPv6 hop-by-hop option.  
    ICMPv6 参数问题消息**不会**为响应未知的 IPv6 逐跳选项而生成。
#### 6LoWPAN

[](https://github.com/smoltcp-rs/smoltcp#6lowpan)

- Implementation of [RFC6282](https://tools.ietf.org/rfc/rfc6282.txt).  
    [RFC6282](https://tools.ietf.org/rfc/rfc6282.txt) 的实施。
- Fragmentation is supported, as defined in [RFC4944](https://tools.ietf.org/rfc/rfc4944.txt).  
    支持分段，如 [RFC4944](https://tools.ietf.org/rfc/rfc4944.txt) 中所定义。
- UDP header compression/decompression is supported.  
    支持 UDP 标头压缩/解压缩。
- Extension header compression/decompression is supported.  
    支持扩展标头压缩/解压缩。
- Uncompressed IPv6 Extension Headers are **not** supported.  
    **不支持**未压缩的 IPv6 扩展标头。
### IP multicast  IP 组播

[](https://github.com/smoltcp-rs/smoltcp#ip-multicast)

#### IGMP

[](https://github.com/smoltcp-rs/smoltcp#igmp)

The IGMPv1 and IGMPv2 protocols are supported, and IPv4 multicast is available.  
支持 IGMPv1 和 IGMPv2 协议，并支持 IPv4 组播。

- Membership reports are sent in response to membership queries at equal intervals equal to the maximum response time divided by the number of groups to be reported.  
    成员资格报告以相等的时间间隔发送以响应成员资格查询，该间隔等于最大响应时间除以要报告的组数。
### ICMP layer  ICMP 层

[](https://github.com/smoltcp-rs/smoltcp#icmp-layer)

#### ICMPv4  ICMPv4 协议

[](https://github.com/smoltcp-rs/smoltcp#icmpv4)

The ICMPv4 protocol is supported, and ICMP sockets are available.  
支持 ICMPv4 协议，并且提供 ICMP 套接字。

- ICMPv4 header checksum is supported.  
    支持 ICMPv4 标头校验和。
- ICMPv4 echo replies are generated in response to echo requests.  
    ICMPv4 回应回复是为响应回应请求而生成的。
- ICMP sockets can listen to ICMPv4 Port Unreachable messages, or any ICMPv4 messages with a given IPv4 identifier field.  
    ICMP 套接字可以侦听 ICMPv4 端口无法访问消息，或具有给定 IPv4 标识符字段的任何 ICMPv4 消息。
- ICMPv4 protocol unreachable messages are **not** passed to higher layers when received.  
    ICMPv4 协议不可达消息在收到时**不会**传递到更高层。
- ICMPv4 parameter problem messages are **not** generated.  
    **不会**生成 ICMPv4 参数问题消息。
#### ICMPv6  ICMPv6 协议

[](https://github.com/smoltcp-rs/smoltcp#icmpv6)

The ICMPv6 protocol is supported, and ICMP sockets are available.  
支持 ICMPv6 协议，并且提供 ICMP 套接字。

- ICMPv6 header checksum is supported.  
    支持 ICMPv6 标头校验和。
- ICMPv6 echo replies are generated in response to echo requests.  
    ICMPv6 回应回复是为响应回应请求而生成的。
- ICMPv6 protocol unreachable messages are **not** passed to higher layers when received.  
    ICMPv6 协议不可达消息在收到时**不会**传递到更高层。
#### NDISC

[](https://github.com/smoltcp-rs/smoltcp#ndisc)

- Neighbor Advertisement messages are generated in response to Neighbor Solicitations.  
    邻居通告消息是为响应邻居请求而生成的。
- Router Advertisement messages are **not** generated or read.  
    **不**生成或不读取路由器播发消息。
- Router Solicitation messages are **not** generated or read.  
    **不**生成或不读取 Router Solicitation 消息。
- Redirected Header messages are **not** generated or read.  
    **重定向**的 Header 消息不会生成或读取。
### UDP layer  UDP 层

[](https://github.com/smoltcp-rs/smoltcp#udp-layer)

The UDP protocol is supported over IPv4 and IPv6, and UDP sockets are available.  
IPv4 和 IPv6 支持 UDP 协议，并且 UDP 套接字可用。

- Header checksum is always generated and validated.  
    始终生成和验证标头校验和。
- In response to a packet arriving at a port without a listening socket, an ICMP destination unreachable message is generated.  
为了响应到达没有侦听套接字的端口的数据包，将生成 ICMP destination unreachable 消息。
### TCP layer  TCP 层

[](https://github.com/smoltcp-rs/smoltcp#tcp-layer)

The TCP protocol is supported over IPv4 and IPv6, and server and client TCP sockets are available.  
IPv4 和 IPv6 支持 TCP 协议，并且提供服务器和客户端 TCP 套接字。

- Header checksum is generated and validated.  
    生成并验证标头校验和。
- Maximum segment size is negotiated.  
    最大区段大小是协商的。
- Window scaling is negotiated.  
    窗口缩放是协商的。
- Multiple packets are transmitted without waiting for an acknowledgement.  
    传输多个数据包，而无需等待确认。
- Reassembly of out-of-order segments is supported, with no more than 4 or 32 gaps in sequence space.  
    支持乱序片段的重组，序列空间中的间隙不超过 4 或 32 个。
- Keep-alive packets may be sent at a configurable interval.  
    可以按可配置的间隔发送 keep-alive 数据包。
- Retransmission timeout starts at at an estimate of RTT, and doubles every time.  
    重传超时从 RTT 的估计值开始，每次都会翻倍。
- Time-wait timeout has a fixed interval of 10 s.  
    时间等待超时的固定间隔为 10 秒。
- User timeout has a configurable interval.  
    用户超时具有可配置的间隔。
- Delayed acknowledgements are supported, with configurable delay.  
    支持延迟确认，具有可配置的延迟。
- Nagle's algorithm is implemented.  
    实现了 Nagle 的算法。
- Selective acknowledgements are **not** implemented.  
    **不**实施选择性确认。
- Silly window syndrome avoidance is **not** implemented.  
    **Silly** Window 综合症避免没有实施。
- Congestion control is **not** implemented.  
    **未**实施拥塞控制。
- Timestamping is **not** supported.  
    **不支持**时间戳。
- Urgent pointer is **ignored**.  
    紧急指针被**忽略** 。
- Probing Zero Windows is **not** implemented.  
    **未**实现探测零窗口。
- Packetization Layer Path MTU Discovery [PLPMTU](https://tools.ietf.org/rfc/rfc4821.txt) is **not** implemented.  
    **未**实施分组层路径 MTU 发现 [PLPMTU](https://tools.ietf.org/rfc/rfc4821.txt)。