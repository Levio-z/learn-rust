---
tags:
  - note
---
## 1. 核心观点  

我们如何安排`wake` 等插座可读后再调用吗？一种选择是 一个持续检查`socket`是否可读的线程，调用 在适当情况下，`wake（）` 然而，这样做效率较低，需要为每个阻塞的 IO 未来单独设置线程。这会大大降低我们异步代码的效率。

实际上，这个问题通过集成 IO 感知的阻断原语来解决，例如 Linux 上的 `epoll`、FreeBSD 和 Mac OS 上的 `kqueue`、Windows 上的 IOCP，以及 Fuchsia 上的 `port`s（这些都通过跨平台的 Rust crate [`mio`](https://github.com/tokio-rs/mio) 暴露）。这些原语都允许线程在多个异步 IO 事件中阻塞，并在其中一个事件完成后返回。实际上，这些 API 通常大致如下：
```rust
struct IoBlocker {
    /* ... */
}

struct Event {
    // 唯一标识已发生并被监听事件的 ID。
    id: usize,

    // 一组等待中或已发生的信号（如可读、可写）。
    signals: Signals,
}

impl IoBlocker {
    /// 创建一个新的异步 IO 事件集合用于阻塞监听。
    fn new() -> Self { /* ... */ }

    /// 注册对特定 IO 事件的兴趣。
    fn add_io_event_interest(
        &self,

        /// 将在其上发生事件的对象（如 Socket）
        io_object: &IoObject,

        /// 一组可能在 `io_object` 上出现的信号，用于触发事件；
        /// 同时配对一个 ID，以便为该兴趣产生的事件命名/识别。
        event: Event,
    ) { /* ... */ }

    /// 阻塞当前线程，直到注册的事件之一发生。
    fn block(&self) -> Event { /* ... */ }
}

// --- 使用示例 ---

let mut io_blocker = IoBlocker::new();

// 注册兴趣：监听 socket_1 的可读信号，绑定 ID 为 1
io_blocker.add_io_event_interest(
    &socket_1,
    Event { id: 1, signals: READABLE },
);

// 注册兴趣：监听 socket_2 的可读或可写信号，绑定 ID 为 2
io_blocker.add_io_event_interest(
    &socket_2,
    Event { id: 2, signals: READABLE | WRITABLE },
);

// 线程在此阻塞，等待操作系统通知
let event = io_blocker.block();

// 如果 socket_1 变得可读，则打印类似 "Socket 1 is now READABLE"
println!("Socket {:?} 现在状态为 {:?}", event.id, event.signals);
```

Futures 执行器可以使用这些原语提供异步 IO 对象，如套接字，这些对象可以配置回调，在特定 IO 事件发生时执行。以我们上面的 `SocketRead` 例子为例， `Socket：：set_readable_callback` 函数可能看起来像以下伪代码：

```rust
impl Socket {
    fn set_readable_callback(&self, waker: Waker) {
        // `local_executor` 是对本地执行器的引用。
        // 这个引用可以在创建 Socket 时提供，但在实践中，
        // 许多执行器实现为了方便，会通过线程局部存储（Thread Local Storage）来传递它。
        let local_executor = self.local_executor;

        // 此 IO 对象的唯一 ID。
        let id = self.id;

        // 将本地 waker（唤醒器）存储在执行器的映射表中，
        // 以便在 IO 事件到达时能够调用它。
        local_executor.event_map.insert(id, waker);

        // 告诉执行器：我们现在对这个文件描述符的“可读”事件感兴趣。
        local_executor.add_io_event_interest(
            &self.socket_file_descriptor,
            Event { id, signals: READABLE },
        );
    }
}
```
我们现在只需一个执行线程，可以接收并派遣任何 IO 事件到相应的 `Waker`，唤醒相应任务，使执行者能够驱动更多任务完成，然后返回检查更多 IO 事件（循环继续......）。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-执行器-基本概念](Rust-future-执行器-基本概念.md)
	- [Rust-future-执行器-基本概念](Rust-future-执行器-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
