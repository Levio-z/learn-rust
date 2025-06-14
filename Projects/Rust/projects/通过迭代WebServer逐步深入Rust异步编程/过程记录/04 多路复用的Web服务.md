现在我们的 Web 服务能在单线程中执行多个请求，没有任何阻塞。如果某个操作被阻塞，它将记住自己的状态并切换，让其他操作执行，这跟内核调度器的行为一致。但是，新设计带来了两个问题.

首先是**所有的工作都在主线程中运行，只利用了一个 CPU 核心**。我们尽最大努力高效地利用这一核心，但一次仍然只执行一个任务。如果线程能分布在多个核心上，我们同一时间就可以做更多的工作。
不过有一个更大的问题。

我们的主循环效率并不高。
**我们对每一个活跃的连接，每一次循环的迭代，都要向内核发出一个 I/O 请求，来检查它是否准备好了。即使调用 read 或 write 返回了 WouldBlock，实际没有执行任何 I/O，它仍然是一个系统调用。系统调用并不便宜**。我们可能有 10k 个活跃的连接，但只有 500 个是准备好的。当只有 500 个连接会真正做些事情的时候，调用 read 或 write 10k 次是对 CPU 周期的巨大浪费。

随着连接数的增加，我们的循环变得越来越低效，浪费了更多的时间做无用的工作。

怎么解决这个问题呢？使用阻塞 I/O 时，内核能够有效地调度任务，因为它知道资源什么时候准备好了。使用非阻塞 I/O 时，我们不检查就不知道，但是检查是很昂贵的。

我们需要的是一种高效的方式来跟踪所有的活跃连接，并且在它们准备好的时候得到通知。

事实证明，我们并不是第一个遇到这个问题的人。每个操作系统都提供了针对这个问题的解决方案。在 Linux 上，它叫做 epoll。

epoll(7) - I/O 事件通知机制

epoll API 执行的任务与 poll(2) 类似：监视多个文件描述符，看看是否有任何一个可以进行 I/O 操作。epoll API 可以作为边缘触发（edge-triggered）或水平触发（level-triggered）的接口使用，并且能够很好地扩展到监视大量的文件描述符。

听上去很完美！我们试试看。

epoll 是一组 Linux 系统调用，**让我们可以处理一组非阻塞的套接字。直接使用 epoll 并不是很方便，所以我们将使用 epoll crate，它是一个对 C 接口的轻度封装。**

首先，我们使用 create 函数来初始化一个 epoll 实例。

```rust
// ```toml
// [dependencies]
// epoll = "4.3"
// ```

fn main() {
    let epoll = epoll::create(false).unwrap(); // 👈
}


```
epoll::create 返回一个文件描述符，它代表了新创建的 epoll 实例。你可以把它看作是一个文件描述符的集合，我们可以从中添加或删除文件描述符。

在 Linux/Unix 中，一切都被视为文件。文件系统上的实际文件、TCP 套接字、以及外部设备都是可以读写的文件。文件描述符是一个整数，它表示系统中打开的“文件”。本文接下来的部分，我们将频繁使用它。

我们要添加的第一个文件描述符是 TCP 监听器。可以用 epoll::ctl 命令来修改 epoll 集合，添加文件描述符使用EPOLL_CTL_ADD标志。

```rust
use epoll::{Event, Events, ControlOptions::*};
use std::os::fd::AsRawFd;

fn main() {
    let listener = TcpListener::bind("localhost:3000").unwrap();
    listener.set_nonblocking(true).unwrap();

    // 添加 listener 到 epoll
    let event = Event::new(Events::EPOLLIN, listener.as_raw_fd() as _);
    epoll::ctl(epoll, EPOLL_CTL_ADD, listener.as_raw_fd(), event).unwrap(); // 👈
}


```
我们传入要注册的资源的文件描述符，也就是 TCP 监听器，以及一个事件。一个事件有两个部分，interest flag 和 data field。interest flag 让我们可以告诉 epoll 我们感兴趣的 I/O 事件。在 TCP 监听器中，我们想要在有新连接进来时得到通知，所以传入 EPOLLIN 标志。
- `Events::EPOLLIN`  
    表示你关注的是**可读事件**（即“有数据可读”或“有新连接”）。
- `as u64`  
    epoll 的事件结构体里有一个 `data` 字段（类型是 `u64`），你可以存任何你想携带的信息，通常存 fd 以便识别事件对应的 socket。
	- epoll 每次返回事件时，只告诉你“哪个事件发生了”，你自己需要根据 `data` 字段判断是哪个 fd 或哪个连接。
	- 通常用 fd 作为 `data`，后续根据 fd 找到对应连接或数据结构。
data field 让我们可以存储一个能够唯一标识每个资源的 ID。记住，文件描述符是一个给定文件的唯一整数，所以直接使用它。你会在下一步看到为什么这很重要。
现在轮到主循环。这次不用自旋，用 epoll::wait。

epoll_wait(2) - 等待 epoll 文件描述符上的 I/O 事件

**对文件描述符 epfd 指向的 epoll(7) 实例而言，epoll_wait() 系统调用会等待其上的事件。interest list 中的文件描述符指向的 ready list 中，如果有一些可用事件的信息，那么这些信息通过 events 指向的缓冲区返回。**

调用 epoll_wait() 将阻塞，直到以下任一情况发生：

- 文件描述符提交了一个事件；
- 调用被信号处理器中断；
- 超时；

epoll::wait 是 epoll 的神奇之处：**它阻塞直到我们注册的任何事件变得就绪，并告诉我们哪些事件就绪了。此时这仅用于有新连接进来时，但是很快我们将使用它来阻塞读、写和刷新事件，这些事件我们之前是用自旋的方式处理的。**

您可能不喜欢 epoll::wait 是“阻塞”的这一事实，但是，它只在没有任何事情可做的时候才阻塞，而之前我们是在自旋并且做无用的系统调用。这种同时阻塞多个操作的方法被称为 _I/O 多路复用_。

**epoll::wait 接受一个事件的列表，当所关注的文件描述符就绪，它会将文件描述符的信息填充到列表，然后返回被添加的事件的数量。**
```rust
// ...
loop {
    let mut events = [Event::new(Events::empty(), 0); 1024];
    let timeout = -1; // 阻塞，直到某些事情发生
    let num_events = epoll::wait(epoll, timeout, &mut events).unwrap(); // 👈

    for event in &events[..num_events] {
        // ...
    }
}


```
epoll::wait 行为解释
- `epoll::wait(epoll, timeout, &mut events)` 的作用是：
    - 等待 epoll 实例 `epoll` 上注册的文件描述符（fd）有事件发生。
    - 当有 fd 就绪时，**把这些就绪事件的详细信息**填入 `events` 数组（你这里是长度 1024 的数组）。
    - 返回值 `num_events` 是**本次被触发的事件个数**。
参数详解
- `epoll`：epoll 实例的句柄（fd）。
- `timeout`：  `超时` ：
    - `-1` 表示**一直阻塞**，直到有事件发生。
    - `0` 表示**立即返回**，不会阻塞（即“轮询”）。
    - `>0` 表示**等待指定毫秒数**，超时后就算没事件也会返回。
- `&mut events`：用于接收事件的数组。
事件数组
- 每个 `Event` 结构体包含了：
    - 就绪的事件类型（可读、可写、异常等）
    - 你注册时填进去的 `data` 字段（通常是 fd 或者你的自定义数据）
每个事件都包含数据字段，该字段与就绪的资源相关联。
```rust
for event in &events[..num_events] {
    let fd = event.data as i32;
    // ...
}

```
还记得我们用文件描述符来标记数据字段吗？我们可以用它来检查事件是否是针对TCP监听器的，如果是，那就意味着有一个传入连接已经准备好被接受了。
```rust
for event in &events[..num_events] {
    let fd = event.data as i32;

    // listener准备就绪了吗?
    if fd == listener.as_raw_fd() {
        // 尝试建立一个连接
        match listener.accept() {
            Ok((connection, _)) => {
                connection.set_nonblocking(true).unwrap();

                // ...
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("{e}"),
        }
    }
}

```
如果返回 WouldBlock， 则移动到下一个连接，等待下一次事件发生。

现在需要在 epoll 中注册新的连接，跟注册侦听器一样。
```rust
for event in &events[..num_events] {
    let fd = event.data as i32;

    // listener准备就绪了吗?
    if fd == listener.as_raw_fd() {
        // 尝试建立一个连接
        match listener.accept() {
            Ok((connection, _)) => {
                connection.set_nonblocking(true).unwrap();
                 let fd = connection.as_raw_fd();

                 // 注册新连接到epoll
                 let event = Event::new(Events::EPOLLIN | Events::EPOLLOUT, fd as _);
                 epoll::ctl(epoll, EPOLL_CTL_ADD, fd, event).unwrap(); // 👈
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("{e}"),
        }
    }
}

```
这次我们注册了EPOLLIN 和 EPOLLOUT 事件，因为根据连接状态，我们要关注读或写事件。

注册了连接之后，我们将得到 TCP 侦听器和某个连接的事件。我们需要用某种方式存储连接和它们的状态，并能通过查找文件描述符的方式来访问它们。

这次不用 List，用 HashMap。
```rust
let mut connections = HashMap::new();

loop {
    // ...
    'next: for event in &events[..num_events] {
        let fd = event.data as i32;

        // listener准备就绪了吗?
        if fd == listener.as_raw_fd() {
            match listener.accept() {
                Ok((connection, _)) => {
                    // ...

                    let state = ConnectionState::Read {
                        request: [0u8; 1024],
                        read: 0,
                    };

                    connections.insert(fd, (connection, state)); // 👈
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("{e}"),
            }

            continue 'next;
        }

        // listener未就绪的话，必须有连接是就绪的
        let (connection, state) = connections.get_mut(&fd).unwrap(); // 👈
    }
}

```
一旦连接和它的状态就绪，我们可以用和之前一样的方法来推进它。**从流中读写数据的操作没有任何变化，区别是现在我们仅在接到 epoll 通知时才进行操作。**
- 通常在 epoll 事件循环中，当某个 fd（不是 listener，而是已建立连接的 fd）有事件时，从 `connections` 取出对应的连接对象和状态，进行读写或状态机处理。

以前我们必须检查每一个连接，看看是否有什么变得就绪，但现在由 epoll 来处理，避免了任何无用的系统调用。
```rust
// ...

// epoll告诉我们连接是否就绪
let (connection, state) = connections.get_mut(&fd).unwrap();

if let ConnectionState::Read { request, read } = state {
    // connection.read...
    *state = ConnectionState::Write { response, written };
}

if let ConnectionState::Write { response, written } = state {
    // connection.write...
    *state = ConnectionState::Flush;
}

if let ConnectionState::Flush = state {
    // connection.flush...
}


```
所有操作都完成后，我们从 connections 中移除当前连接，它会自动从 epoll 中注销。
```rust
for fd in completed {
    let (connection, _state) = connections.remove(&fd).unwrap();
    // 会自动从epoll注销
    drop(connection);
}

```
- 假如你用 Rust 的 `TcpStream` 作为连接对象，`TcpStream` 在 drop 时会自动 close fd