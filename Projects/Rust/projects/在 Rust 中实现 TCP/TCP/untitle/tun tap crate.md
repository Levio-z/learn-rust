### 方法
#### recv
#### pub fn [recv](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.recv)(&self, buf: &mut [[u8](https://doc.rust-lang.org/nightly/std/primitive.u8.html)]) -> [Result](https://doc.rust-lang.org/nightly/std/io/error/type.Result.html "type std::io::error::Result")<[usize](https://doc.rust-lang.org/nightly/std/primitive.usize.html)>  
pub fn[recv](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.recv)（self，buf：mut [[u8](https://doc.rust-lang.org/nightly/std/primitive.u8.html)]）-> [结果](https://doc.rust-lang.org/nightly/std/io/error/type.Result.html "type std::io::error::Result") < [大小](https://doc.rust-lang.org/nightly/std/primitive.usize.html) >

Receives a packet from the interface.  
从接口接收数据包。

By default, blocks until a packet is sent into the virtual interface. At that point, the content of the packet is copied into the provided buffer.  
默认情况下，会一直阻塞，直到数据包发送到虚拟接口。在这一点上，数据包的内容被复制到提供的缓冲区。

If interface has been set to be non-blocking, this will fail with an error of kind [`WouldBlock`](https://doc.rust-lang.org/nightly/std/io/error/enum.ErrorKind.html#variant.WouldBlock "variant std::io::error::ErrorKind::WouldBlock") instead of blocking if no packet is queued up.  
如果接口已设置为非阻塞，则此操作将失败，并返回一个类型错误 [](https://doc.rust-lang.org/nightly/std/io/error/enum.ErrorKind.html#variant.WouldBlock "variant std::io::error::ErrorKind::WouldBlock")如果没有数据包排队，将阻塞而不是阻塞。

Make sure the buffer is large enough. It is MTU of the interface (usually 1500, unless reconfigured) + 4 for the header in case that packet info is prepended, MTU + size of ethernet frame (38 bytes, unless VLan tags are enabled). If the buffer isn’t large enough, the packet gets truncated.  
确保缓冲区足够大。它是接口的 MTU（通常为 1500，除非重新配置）+ 4（如果数据包信息被预先添加），MTU +以太网帧的大小（38 字节，除非启用 VLAN 标签）。如果缓冲区不够大，数据包就会被截断。
#### new
#### pub fn [new](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.new)(ifname: &[str](https://doc.rust-lang.org/nightly/std/primitive.str.html), mode: [Mode](https://docs.rs/tun-tap/latest/tun_tap/enum.Mode.html "enum tun_tap::Mode")) -> [Result](https://doc.rust-lang.org/nightly/std/io/error/type.Result.html "type std::io::error::Result")<Self>  
pub fn[new](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.new)（ifname：[str](https://doc.rust-lang.org/nightly/std/primitive.str.html)，mode：[Mode](https://docs.rs/tun-tap/latest/tun_tap/enum.Mode.html "enum tun_tap::Mode")）-> [结果](https://doc.rust-lang.org/nightly/std/io/error/type.Result.html "type std::io::error::Result")

Creates a new virtual interface.  
创建新的虚拟接口。

##### Parameters  参数

- `ifname`: The requested name of the virtual device. If left empty, the kernel will provide some reasonable, currently unused name. It also can contain `%d`, which will be replaced by a number to ensure the name is unused. Even if it isn’t empty or doesn’t contain `%d`, the actual name may be different (for example truncated to OS-dependent length). Use [`name`](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.name) to find out the real name.  
    `ifname`：虚拟设备的请求名称。如果保留为空，内核将提供一些合理的、当前未使用的名称。它也可以包含 `%d`，这将被替换为一个数字，以确保名称未使用。即使它不为空或不包含 `%d`，实际名称也可能不同（例如截断为操作系统相关的长度）。用 [`name`](https://docs.rs/tun-tap/latest/tun_tap/struct.Iface.html#method.name) 找出真实的名字。
- `mode`: In which mode to create the device.  
    `mode`：创建设备的模式。