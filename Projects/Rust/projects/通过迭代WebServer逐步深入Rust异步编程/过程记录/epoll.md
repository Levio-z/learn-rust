## 1. epoll 与 poll/select 区别

- **select/poll**：每次调用都要把所有fd（文件描述符）从用户态拷贝到内核态，遍历一遍，效率低，fd数量多时非常慢。
- **epoll**：把关注的fd集合注册到内核，只需一次，后续只需等待内核通知，有事件才收集，效率高，适合大批量fd。

---

## 2. epoll 支持两种触发模式

### 水平触发（Level-triggered, LT）

- **原理**：只要 fd 上有数据可读/可写，`epoll_wait` 每次都会通知你。
- **行为**：即使你这次没把数据全读走，下次 `epoll_wait` 还会提醒你“还有东西没读”。
- **默认模式**：如果你不指定 `EPOLLET` 标志，epoll 就是 LT 模式。
- **适用场景**：和 select/poll 行为一致，逻辑简单，适合新手。

### 边缘触发（Edge-triggered, ET）

- **原理**：只有当 fd 的 I/O 状态发生变化时才通知你，比如“从无数据到有数据”。
- **行为**：如果你没把数据一次读完，下次 `epoll_wait` 就不会再提醒你，**会漏事件**。
- **高效**：减少事件通知次数，适合高性能场景。
- **使用要求**：通常配合非阻塞 fd，循环读/写直到返回 `EAGAIN`（即读/写完为止）。
- **如何开启**：注册事件时加上 `EPOLLET` 标志即可。

#### 例子对比

- LT：每次有数据都通知你，数据没读完还会再通知。
- ET：只有新数据到达才通知你，数据没读完不再通知，必须全部读完。
---

## 3. epoll 主要 API

- `epoll_create` / `epoll_create1`：创建epoll实例
- `epoll_ctl`：注册/修改/删除感兴趣的fd
- `epoll_wait`：等待事件发生

1. **epoll_create/epoll_create1**
    - 创建 epoll 实例，返回一个 epoll 文件描述符。
2. **epoll_ctl**
    - 注册、修改、移除你想关注的 fd 及其事件类型（读/写/ET）。
    - 常用操作：
        - `EPOLL_CTL_ADD`：添加监听
        - `EPOLL_CTL_MOD`：修改监听
        - `EPOLL_CTL_DEL`：移除监听
3. **epoll_wait**
    - 阻塞等待事件发生，返回已就绪的 fd 列表。
---

## 4. 适用场景

- 网络服务器、聊天室、消息中间件、高并发代理等
- 大量连接，少量活跃连接，I/O密集型

---

## 5. Rust/其他语言的 epoll 封装

- Rust常用库如 [mio](https://github.com/tokio-rs/mio)、tokio等，底层也是用epoll实现的高效异步I/O。

---

如需详细示例、边缘触发与水平触发的代码对比，或实现原理解析，可以继续提问！

### 常见的 epoll 事件标志

|标志|说明|
|---|---|
|`EPOLLIN`|读事件：有数据可读，或者有新连接（对于监听 socket）|
|`EPOLLOUT`|写事件：写操作不会被阻塞（可以写入数据）|
|`EPOLLRDHUP`|对端关闭连接，或者半关闭（常用于检测对端断开）|
|`EPOLLPRI`|有紧急数据可读（带外数据，极少用）|
|`EPOLLERR`|发生错误（如连接异常断开）|
|`EPOLLHUP`|挂断事件（对端关闭、管道写端关闭等）|
|`EPOLLET`|边缘触发模式（Edge Triggered），更高效但编程更复杂|
|`EPOLLONESHOT`|只监听一次事件，事件触发后自动从 epoll 中移除（需手动重新注册）|
|`EPOLLWAKEUP`|唤醒系统（用于 power management，极少用）|
|`EPOLLEXCLUSIVE`|互斥唤醒（避免惊群，通常在多线程高并发场景下监听同一个 fd）|
## 组合方式

- 这些标志可以用按位或 `|` 组合，比如：
    
    Rust  锈
    
    ```
    Events::EPOLLIN | Events::EPOLLET
    ```
    
    表示关注“可读事件”且采用“边缘触发”模式。
### Rust epoll crate 常用事件举例

Rust  锈

```
use epoll::Events;

// 可读
Events::EPOLLIN
// 可写
Events::EPOLLOUT
// 错误
Events::EPOLLERR
// 挂断
Events::EPOLLHUP
// 边缘触发
Events::EPOLLET
// 一次性触发
Events::EPOLLONESHOT
```