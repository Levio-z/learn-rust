https://github.com/Centaurus99/arceos-lwip/tree/main/reports

#1
- 分析 ArceOS 的 C app 运行方式
	- `libax_bindings` 对 `rust_libax` 做了一层 C 接口的包装，并通过 `cbindgen` 生成头文件
	- C app 的 include 目录为 `ulib/c_libax/include` 和 `ulib/c_libax/libax_bindings`
	- 通过指定 features 来控制 `libax` 中使用的 modules，从而实现组件化
	- 从 [musl.cc](https://musl.cc/) 下载交叉编译工具链，尝试编译运行了三个 C app
- 分析 smoltcp 的使用
	- `axnet` 通过 `smoltcp_impl` 对 `smoltcp` 进行包装，对下使用 `axdriver::NetDevices`，`driver_net` 等模块驱动网卡，对上提供 `TcpSocket` 以供应用使用 TCP
	- 实验的一大目标即为实现 `lwip_impl` 来对 C 实现的 `lwip` 进行包装与适配
- 初识 LwIP
	- 纯 C 实现，对内核中会使用到操作系统功能的地方进行了抽象，可以通过实现这些API完成迁移
	- 支持协议较为完整
	- 实现了一些常见的应用程序，完成迁移后或许都可以尝试跑起来
#2
- axruntime` 在初始化时调用 `axdriver::init_drivers` 来初始化各个设备和驱动（每种类型的设备目前似乎只能有一个？）
- 具体的驱动实现在外部模块 [virtio-drivers](https://github.com/rcore-os/virtio-drivers) 中。对于 net，初始化时会创建 `send_queue` 和 `recv_queue` 这两个 DMA Descriptor Ring，默认长度 64，同时为 `recv_queue` 中的每个 descriptor 创建 buffer（`rx_buffers`），buffer 默认长度 2048。然后包装为 `VirtIoNetDev` 并实现了 trait `NetDriverOps` 供上层使用。
- 接着通过 `axnet::init_network` 对网卡做一些初始化。`axnet` 将 `VirtIoNetDev` 包装为 `DeviceWrapper`，并对其实现了 smoltcp 的 `Device` trait，供 smoltcp 控制设备收发以太网帧。在 receive 时返回 `AxNetRxToken` 和 `AxNetTxToken`，transmit 时返回 `AxNetTxToken`。在 `RxToken` 被 consume 时，将 `RxToken` 里对应 buffer 的包做相应处理，然后回收进网卡驱动中的收包队列；在 `TxToken` 被 consume 时，让网卡驱动创建发包的 buffer，然后向其中填充数据并发送。
#3
- 本周初步完成 NO_SYS 模式的 lwip 移植。
- 分析 lwip
- 移植 lwip 分析
- 移植 lwip 实操
#4
- 初步完成上层接口适配，代码量约 400 行
#5
- 对上游仓库的修改进行适配，如添加 UDP 接口等（暂未做功能实现）。
- 异步模型
#6 
- 更换开发平台
- 配置 TAP  放置 TAP
- 系统层适配初探
#7
- 进行 UDP 和 DNS 适配。
- 学习 lwip 的 UDP raw api：
- 学习 lwip 的 DNS raw api：
#8
- 学习 `smoltcp_impl` 的适配方式，进行适配。
- ci多架构的适配
#9
- 链接时的浮点问题
- 移除 `c_libax` 依赖
- 使用 `APP_FEATURES=libax/use-lwip` 启用 lwip 网络栈
#10
- 修 bug
