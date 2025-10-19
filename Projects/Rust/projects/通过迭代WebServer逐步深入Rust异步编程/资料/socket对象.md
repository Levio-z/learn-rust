### 先说本质定义：什么是 socket？

> **Socket（套接字）** 是一种**操作系统提供的抽象通信接口**，用于支持**进程间或网络间通信**。

| 通信类型           | 描述                 | 示例              |
| -------------- | ------------------ | --------------- |
| **进程间通信（IPC）** | 同一台机器中，不同进程之间的数据交换 | Chrome 的多个进程间通信 |
| **网络通信**       | 跨机器之间，通过网络协议进行通信   | 客户端请求服          |

它不是 Rust 或 Java 的发明，而是 Unix/Linux 网络编程的底层基础概念。

简言之，**Socket 是内核中的通信端点**，具备以下信息：
- 协议类型（TCP/UDP）
- 本地地址（IP:PORT）
- 对端地址（IP:PORT）
- 状态（连接中、已连接、关闭）
- 缓冲区（收发数据）
- 读写操作接口（read、write、send、recv）
- 你可以把 socket 想象成“网络版的文件句柄”，只是它是“通向远方进程的管道”。
### Rust 中的 socket 对象：`TcpStream`、`TcpListener`
在 Rust 中，我们通过 `std::net` 模块使用 socket 的**封装对象**。例如：
- `TcpListener`：**监听套接字**，代表一个“正在监听端口”的服务器 socket。
- `TcpStream`：**连接套接字**，代表一个“已建立连接”的 socket，用于双向读写数据。
- 此处：
	- `listener` 是 **监听 socket**（底层系统 socket，Rust 封装为 TcpListener）。
	- `stream` 是 **连接 socket**（accept 成功返回的 socket，Rust 封装为 TcpStream）。
### 底层工作机制（简化版）
```rust
监听 socket（fd: 3） ← 监听 0.0.0.0:8080
                 ↓
           客户端连接进来
                 ↓
    accept() 系统调用生成新的 socket（fd: 4）
                 ↓
       返回 TcpStream(fd: 4) 给应用程序


```
所以：
- `TcpListener` 内部是一个监听 socket 文件描述符（如 fd=3）
- 每次 `accept()` 调用，**内核生成一个新的 socket（如 fd=4）**
- `TcpStream` 封装了这个“新 socket”（fd=4）
## 类比理解

| 类别          | Socket                          | TcpStream / TcpListener（Rust 对象）    |
| ----------- | ------------------------------- | ----------------------------------- |
| 层级          | 操作系统内核对象                        | 应用层对象（封装了内核 socket）                 |
| 类型          | 监听 socket / 连接 socket           | TcpListener / TcpStream             |
| 创建方式        | `socket()` / `accept()` syscall | `TcpListener::bind()` / `.accept()` |
| 功能          | 网络通信的端点                         | 读写（Stream）或接收连接（Listener）           |
| Rust 是否直接操作 | 否，必须通过封装类型使用                    | 是，Rust 提供了安全封装                      |
**Socket 是系统级通信接口，`TcpStream` 和 `TcpListener` 是 Rust 中对其的封装对象**，用于进行 TCP 通信。

- `listener` 是服务器监听 socket 的封装（`TcpListener`）
- `accept()` 是一个**阻塞式的系统调用**
- 它返回一个新的 **连接 socket**（封装成 `TcpStream`）

### **服务器监听 socket**
是服务器进程在指定 IP 和端口上创建的特殊 socket，它用于**等待客户端发起 TCP 连接请求**，并为每个请求生成新的连接 socket。

在 Linux/Unix 操作系统中，服务器监听 socket 的创建涉及以下系统调用（Rust 中是封装好了的）：

|步骤|系统调用|作用|
|---|---|---|
|①|`socket()`|创建一个 socket 文件描述符（fd）|
|②|`bind()`|将该 socket 绑定到本地的 IP 和端口|
|③|`listen()`|将 socket 设为监听状态（准备接受连接）|
|④|`accept()`|阻塞等待客户端连接，成功后返回一个新 socket（用于通信）|
#### 服务器监听 socket 的作用

- **监听连接请求**：它就像一个“大门口”，永远监听有没有客户端来“敲门”。
- **不是用来读写数据的**：监听 socket 自己不能进行数据收发，它的唯一任务是“等连接”，成功后会生成新的连接 socket。
- **生成连接 socket（`TcpStream`）**：每次客户端连接，它调用 `accept()`，生成一个新的 socket，用于读写。