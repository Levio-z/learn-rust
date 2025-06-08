```rust
let listener = TcpListener::bind("localhost:3000").unwrap();

loop {
    let (connection, _) = listener.accept().unwrap();

    if let Err(e) = handle_connection(connection) {
        // ...
    }
}


```
看到问题了吗？
我们一次只能响应一个请求。
1. `listener.accept()` 是阻塞调用
	- 它会阻塞程序直到有一个新的客户端连接到来，然后返回这个连接（`TcpStream`）。
	- 在你的代码里，accept 返回后，马上调用 `handle_connection(connection)` 来处理这个连接。
2. `handle_connection(connection)` 是同步且阻塞的
	- 你的 `handle_connection` 代码会一直执行，处理完当前连接请求后才会返回。
	- 这意味着：当 `handle_connection` 正在处理第一个连接时，主线程**不会**调用 `listener.accept()` 去接收新连接。
- 新连接如何处理？
	- TCP 层会把新的连接请求放入内核的监听队列（backlog）中。
	- 队列长度有限（一般默认 128），如果队列满了，客户端的新连接请求会被拒绝或超时。
	- 只有当 `handle_connection` 处理完毕，主线程重新执行 `listener.accept()`，才会从队列取出一个新连接。
结果
- 你的程序**无法同时处理多个连接**。
- 客户端的并发请求会被排队等待。
- 如果请求处理慢，或者高并发时，客户端连接就可能出现超时或失败。

从一个网络连接读写数据不是即时完成的，中间要经过大量的基础设备，比如网关、路由器等等，如果两个用户同时向我们发起请求会发生什么？十个、十万个呢？随用户规模增长，我们的服务会延迟、卡顿，直至不可用，那么如何改进呢？
可能的选择有好几种，但到目前为止，最简单的是线程的方式。为每个请求创建一个线程，就能我们的服务能响应无限的用户增长，对吧？
```rust
fn main() {
    // ...
    loop {
        let (connection, _) = listener.accept().unwrap();

        // 为每个传入请求创建一个线程
        std::thread::spawn(|| {
            if let Err(e) = handle_connection(connection) {
                // ...
            }
        });
    }
}


```
实际上，这很有可能！就算不是无限的，但随着每个请求都在单独的线程中处理，我们服务的吞吐量会显著增加。
这到底是怎么回事？

跟大多数现代操作系统一样，在 linux 中，**程序都是在一个单独的进程中运行的**。虽然看起来每个活动程序都是同时运行的，但在物理上，一个 CPU 内核一次只能执行一个任务，或者通过超线程技术同时执行两个。为了让所有的程序都能执行，操作系统内核会不断切换它们，暂停当前正在运行的程序，切换到另一个并运行它，如此往复。这些上下文切换以毫秒为单位发生，形成了感觉上的“并行”。

内核调度器通过在多个内核之间分配工作负载来利用多核。每个核心管理一部分进程[2](https://blog.windeye.top/rust_async/learningrustasyncwithwebserver/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc#%E2%91%A1)，这意味着某些程序可以真正意义上得到并行运行。
```sh
 cpu1 cpu2 cpu3 cpu4
|----|----|----|----|
| p1 | p3 | p5 | p7 |
|    |____|    |____|
|    |    |____|    |
|____| p4 |    | p8 |
|    |    | p6 |____|
| p2 |____|    |    |
|    | p3 |    | p7 |
|    |    |    |    |

```
这种调度类型被称为抢占式多任务调度：内核决定进程运行多长时间被抢占，切换到其他进程。
该模式下，内核确保各独立进程不会访问到其他进程的内存，从而保证各种类型的程序都能得到良好地运行。但是，这使得上下文切换更加昂贵，因为内核在执行上下文切换之前必须刷新内存的某些部分，以确保内存被正确隔离[3](https://blog.windeye.top/rust_async/learningrustasyncwithwebserver/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc#%E2%91%A2)。
线程跟进程类似[4](https://blog.windeye.top/rust_async/learningrustasyncwithwebserver/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc#%E2%91%A3)，区别是线程可以与同一父进程下的线程共享内存，从而实现在同一程序的线程之间共享状态。除此之外线程的调度和进程没有任何区别。
- 必须 **更换页表指针（更新 CR3）**；
- 这会导致 **TLB flush**；
- 造成 **较高的上下文切换开销**。

线程跟进程类似[4](https://blog.windeye.top/rust_async/learningrustasyncwithwebserver/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc#%E2%91%A3)，区别是线程可以与同一父进程下的线程共享内存，从而实现在同一程序的线程之间共享状态。除此之外线程的调度和进程没有任何区别。

对我们的服务而言，1 线程 / 1 请求 模式最关键的问题是我们的服务是 I/O 绑定的。`handle_connection` 执行过程中中绝大部分时间并不是用于计算，而是用于等待，**等待从网络连接中收发数据，等待读、写、刷新等 I/O 阻塞的操作执行完毕**。我们希望的是，发送一个 I/O 请求后，让出控制权给内核，等操作完成后内核再将控制权交回。在此期间，内核可以执行其他程序。

通常情况下，处理**一个网络请求时绝大部分时间都在等待其他任务完成，比如数据库查询或接收 HTTP 请求。**
- 多个工作线程效率高的原因是我们可以利用等待的时间来处理其他请求。

### 总结
**一个网络请求时绝大部分时间都在等待其他任务完成，比如数据库查询或接收 HTTP 请求。等待从网络连接中收发数据，等待读、写、刷新等 I/O 阻塞的操作执行完毕**。