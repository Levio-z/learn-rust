https://github.com/reflyable/arceos-udp/tree/report/report
### 选题
- 支持udp协议，进一步支持dns实现域名解析
- 运行ping之类网络应用
- 支持通过域名访问网站
### 进度
#1
- 熟悉aceros，阅读并运行net app
- 阅读aceros与smoltcp交互部分代码: axnet
	- smoltcp：rust TCP/IP栈，支持TCP，UDP
	- axnet：DeviceWrapper，InterfaceWrapper等包装下层网卡驱动，SocketSetWrapper等包装smoltcp，对上层提供TcpSocket
	- 对udp的支持主要是接入somltcp的udp
- 阅读smoltcp部分接口与结构代码: SocketHandle
#2
- 结合文档基本理解了smoltcp socket层的工作机制与和下层的配合
	- socket层维护收发缓冲区与(tcp)状态
	- interface层将数据包存入/取出socket
- 了解了tcp/udp的socket流程
- 基本理解了已完成的tcp部分的机制与思路
- 简要了解了qemu的网络连接机制
实践
基本完成udp栈的基本功能
- 模仿std::net::UdpSocket  模仿 std::net::UdpSocket
- bind, recv_from, send_to, local_addr  
    绑定、recv_from、send_to local_addr
#3
- 在axnet中接入smoltcp的dns栈
- 阅读、参考**std::net和libc部分源代码**
- 在libax中**调用dns相关接口, 实现dns查询的ToSocketAddrs Trait**
- 增加包括**dns功能的httpclient app**
- **整理、完善代码，增加测例，通过ci，提出pr**
#4
- 为c应用提供类似libc的net接口
- 完善udp/dns部分工作, 修正代码格式, 添加文档, 合并PR
- 阅读c_libax代码, 阅读musl-libc部分代码, 明确clibax net完善方向
- 开始进行clibax net完善
#5
- 参考rCore和学长已有实现, 将clibax中socket集成进fs, 由fs_table统一管理, 完成相关api
- 尽量按照linux标准完善net的c接口的rust部分
	- 已经全部完成: socket, connect, shutdown, recv, send, recvfrom, sendto  
    已经全部完成： socket， connect， shutdown， recv， send， recvfrom， sendto
	- 还需最后一层包装: accept, listen
	- 未开始: getaddrinfo  未开始： getaddrinfo
#6
- 尽量按照linux标准完善net的c接口(包括rust部分和c部分)
	- 已经全部完成: socket, connect, shutdown, recv, send, recvfrom, sendto, accept, listen  
    - 已经全部完成： socket， connect， shutdown， recv， send， recvfrom， sendto， accept， listen
#7 
- 按照linux标准完善net的c接口(包括rust部分和c部分)
- 已经全部完成: socket, connect, shutdown, recv, send, recvfrom, sendto, accept, listen, getaddrinfo
- 实现并完善c语言net app
	- udp server
#8
- 接下来工作: 将iperf3至少一部分功能迁移到arceos, 实现与本地iperf3的互操作
-  阅读iperf3源码
	- 实现未支持的c api: 简单起见仅支持用到的功能
    - select  选择
    - set/getsockopt
    - fcntl
    - 更完善的sprintf
    - 几个浮点数api
- 修改iperf3源码绕过某些api的调用
#9
- 将Iperf3迁移到Arceos，可以通过iperf_api正常使用 基本未更改iperf3源代码，只是注释/用0替换了部分获取硬件信息的函数，将头文件引用修改为clibax，替换了不必要的复杂函数 经测试可以进行server和client的连接，作为server进行udp连接时需要在client指定-l=1300(性能较佳), 大于14xx会导致smoltcp丢弃而收不到udp包。 主要更改:
	-  引入嵌入式printf实现来支持更完全的vsnprintf及snprintf
	- 支持若干基本的浮点数库函数
	- 实现select，setsockopt库函数