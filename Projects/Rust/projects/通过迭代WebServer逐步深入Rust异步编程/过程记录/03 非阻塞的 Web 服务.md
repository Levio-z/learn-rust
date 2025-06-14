看起来多线程已经完全满足我们的需求，并且它使用也很简单，那么为什么我们还要继续呢？

您也许听说过线程很“重”、上下文切换非常“昂贵”等说法，但是现在，这并不准确，现代的服务器能毫不费力地处理上万的线程。

问题在于阻塞 I/O 将程序的控制权完全交给了操作系统内核，在程序执行完成之前，我们没有任何的干预手段可用，这让我们实现某些操作变得非常困难，比如取消操作和选择操作。
假设我们要实现优雅的服务关停操作。当我们按下 ctrl+c，程序不会马上退出，而是立刻停止接受新的连接请求，当前已建立连接的任务会继续执行，直到完成，或者是 30 秒后被强行终止，最后服务才退出。

在阻塞 I/O 模式下，这里的问题是：**我们的 accept 循环会阻塞，直到下一个连接到来。我们可以在新连接请求被接受之前或之后检查 ctrl+c 信号，如果在处理 accept 时信号进来，我们必须等待下一次连接被接受，这期间只有内核拥有程序完全的控制权。**
- 这是一个 **核心问题**。你无法“半路喊停”一个 `accept()`，也不能用一种干净的方式终止它。
- 操作系统内核拥有控制权，你的程序处于**挂起**状态；
	- 想取消操作？只能借助“黑魔法”：如关闭监听 socket、发信号打断系统调用等，**代价高、逻辑复杂**；
```rust
loop {
    // 调用accept之前检查ctrl+c信号
    if got_ctrl_c() {
        break;
    }

    // **如果ctrl+c在这里发生，会出现什么情况?**
    let (connection, _) = listener.accept().unwrap();

    // 在新的连接被接受之前，这不会被调用
    if got_ctrl_c() {
        break;
    }

    std::thread::spawn(|| /* ... */);
}

```
我们想要的是像 match 操作一样，针对 I/O，同时侦听连接请求和 ctrl+c 信号：
```rust
loop {
    // 类似这样...
    match {
        ctrl_c() => {
            break;
        },
        Ok((connection, _)) = listener.accept() => {
            std::thread::spawn(|| ...);
        }
    }
}

```
对于运行时间超过 30 秒的任务，又该怎么处理呢？我们可以设置一个标记让线程停止，那么又该多久检测一次标记呢？我们又回到了老问题：因为 I/O 阻塞导致我们丧失了程序的控制权，除了等它执行完毕，没有好的方式来强制取消一个线程的执行。

这正是线程和阻塞 I/O 令人头疼的地方，因为应用程序的控制权完全交给了内核，导致实现基于事件的逻辑变得非常困难。

某些平台下，可以使用平台特定接口来实现这一点，比如[Unix信号处理机制](https://www.cs.kent.edu/~ruttan/sysprog/lectures/signals.html)。虽然信号处理机制简单，并且在某些场景下工作得很好，但在场景变得复杂的时候，信号处理机制会变得非常繁琐。在本文末尾，我们描述了另一种表达复杂控制流的方法。您可以根据实际情况来挑选合适的方式。


那么，有没有既能执行 I/O，又不用出让控制权给内核的实现方法呢？

实际上，还有另一种实现 I/O 操作的方法，称为非阻塞 I/O( non-bloking I/O )。顾名思义，非阻塞操作永远不会阻塞调用线程，它会立即返回，如果给定的资源不可用，则返回一个错误。

通过将 TCP 侦听器和连接置于非阻塞模式，我们可以切换到非阻塞 I/O 的实现方式。
```rust
let listener = TcpListener::bind("localhost:3000").unwrap();
listener.set_nonblocking(true).unwrap();

loop {
    let (connection, _) = listener.accept().unwrap();
    connection.set_nonblocking(true).unwrap();

    // ...
}

```
非阻塞 I/O 的工作模式有一些不同：如果 I/O 请求不能立即完成，内核将返回一个 WouldBlock 错误代码。尽管被表示为错误代码，但 WouldBlock 并不是真正的错误，它只是意味着当前操作无法立即执行完毕，让我们可以自行决定接下来要做什么。
```rust
use std::io;

// ...
listener.set_nonblocking(true).unwrap();

loop {
    let connection = match listener.accept() {
        Ok((connection, _)) => connection,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            // 操作还不能执行
            // ...
        }
        Err(e) => panic!("{e}"),
    };

    connection.set_nonblocking(true).unwrap();
    // ...
}


```
假设在调用 accept() 之后没有连接请求进来，在阻塞 I/O 模式下，我们只能一直等待新的连接，但现在，WouldBlock 不是将控制权交给内核，而是交回我们手里。
我们的 I/O 终于不阻塞了！但此时我们能做点什么呢？

WouldBlock 是一个临时的状态，意味着在未来某个时刻，当前套接字会准备好用于读或写。所以从技术上讲，我们应该一直等到(作者用了自旋这个单词-spin until)套接字状态变成可用( ready )。
```rust
loop {
    let connection = match listener.accept() {
        Ok((connection, _)) => connection,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            continue; // 👈
        }
        Err(e) => panic!("{e}"),
    };
}

```
但是自旋还不如阻塞，至少阻塞 I/O 模式下，操作系统还可以给其他线程执行的机会。所以我们真正需要的，是为全部任务创建一个有序的调度器，来完成曾经由操作系统来为我们做的事情。

让我们从头回顾一遍：

首先我们创建了一个 TCP 侦听器：
```rust
let listener = TcpListener::bind("localhost:3000").unwrap();

```
然后设置它为非阻塞模式：
```rust
listener.set_nonblocking(true).unwrap();
```
接下来进入主循环，循环中第一件事情是接受一个新的 TCP 连接请求。
```rust
// ...

loop {
    match listener.accept() {
        Ok((connection, _)) => {
            connection.set_nonblocking(true).unwrap();

            // ...
        },
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("{e}"),
    }
}

```
现在，我们不能继续直接为已建立的连接服务，导致其他请求被忽略。我们必须能跟踪所有的活动连接。
```rust
// ...

let mut connections = Vec::new(); // 👈

loop {
    match listener.accept() {
        Ok((connection, _)) => {
            connection.set_nonblocking(true).unwrap();
            connections.push(connection); // 👈
        },
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("{e}"),
    }
}

```
但是我们不能无休止地接受连接请求。当没有操作系统调度的便利时，我们需要在主循环的每一次迭代中，将所有的事情都推进一点点。一旦新的连接请求被接受，我们需要处理所有的活跃连接。

对于每一个连接，我们必须执行任何需要的操作来推进请求的处理，无论是读取请求还是写入响应。

```rust
// ...
loop {
    // 尝试接受新的连接请求
    match listener.accept() {
        // ...
    }

    // 针对活跃的连接进行处理
    for connection in connections.iter_mut() {
        // 🤔
    }
}


```
还记得之前的 handle_connection 功能吗？
```rust
fn handle_connection(mut connection: TcpStream) -> io::Result<()> {
    let mut request = [0u8; 1024];
    let mut read = 0;

    loop {
        let num_bytes = connection.read(&mut request[read..])?;  // 👈
        // ...
    }

    let request = String::from_utf8_lossy(&request[..read]);
    println!("{request}");

    let response = /* ... */;
    let mut written = 0;

    loop {
        let num_bytes = connection.write(&response[written..])?; // 👈

        // ...
    }

    connection.flush().unwrap(); // 👈
}

```
我们需要执行不同的 I/O 操作，比如读、写和刷新。阻塞模式下，代码会按我们写的顺序执行。但现在我们必须面对这样一个事实，在执行 I/O 的任何时候都可能面临WouldBlock，导致当前执行无法取得进展。

同时，我们不能简单地丢掉这个结果去处理下一个活动连接，我们需要跟踪当前连接的状态，方便在下次回来时能从正确的地方继续。

我们设计了一个枚举来保存 handle_connetion 的状态，它有三种可能的状态：
```rust
enum ConnectionState {
    Read,
    Write,
    Flush
}

```
请记住，我们需要的不是记录事务单独的状态，例如将请求转换为字符串，我们需要的是在遇到 WouldBlock 时，能记住当时的状态。

读、写操作的状态还包含当前已读写的字节数和本地缓存。之前我们在函数中定义它，现在我们需要它在整个主循环的生命周期中存在。
```rust
enum ConnectionState {
    Read {
        request: [u8; 1024],
        read: usize
    },
    Write {
        response: &'static [u8],
        written: usize,
    },
    Flush,
}

```
我们在每一次 handle_connection 开始执行时初始化连接状态为 Read，request 为 0 值，read 为 0 字节。
```rust
// ...

let mut connections = Vec::new();

loop {
    match listener.accept() {
        Ok((connection, _)) => {
            connection.set_nonblocking(true).unwrap();


            let state = ConnectionState::Read { // 👈
                request: [0u8; 1024],
                read: 0,
            };

            connections.push((connection, state));
        },
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
        Err(e) => panic!("{e}"),
    }
}

```
现在，我们可以尝试根据其当前状态，将每个连接向前推进了。
```rust
// ...
loop {
    match listener.accept() {
        // ...
    }

    for (connection, state) in connections.iter_mut() {
        if let ConnectionState::Read { request, read } = state {
            // ...
        }

        if let ConnectionState::Write { response, written } = state {
            // ...
        }

        if let ConnectionState::Flush = state {
            // ...
        }
    }
}

```
如果当前连接仍然处于 Read 状态，继续做读取操作，唯一不同的是，如果收到WouldBlock, 则继续处理下一个活动连接。
```rust
// ...

'next: for (connection, state) in connections.iter_mut() {
    if let ConnectionState::Read { request, read } = state {
        loop {
            // 尝试从流中读取数据
            match connection.read(&mut request[*read..]) {
                Ok(n) => {
                    // 跟踪已读取的字节数
                    *read += n
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    // 当前连接的操作还未就绪，继续处理下一个连接
                    continue 'next; // 👈
                }
                Err(e) => panic!("{e}"),
            }

            // 判断是否读到结束标记
            if request.get(*read - 4..*read) == Some(b"\r\n\r\n") {
                break;
            }
        }

        // 操作完成，打印收到的请求数据
        let request = String::from_utf8_lossy(&request[..*read]);
        println!("{request}");
    }

    // ...
}

```
还有读到 0 字节的问题需要处理，之前我们只是从 handle_connection 中退出，state 变量会自动被清空。但是现在，我们必须自己处理当前连接。当前我们正在遍历connections 列表，所以需要一个单独的列表来收集已完成的活动连接，后续再来处理。
```rust
let mut completed = Vec::new(); // 👈

'next: for (i, (connection, state)) in connections.iter_mut().enumerate() {
    if let ConnectionState::Read { request, read } = state {
        loop {
            // 尝试从流中读取数据
            match connection.read(&mut request[*read..]) {
                Ok(0) => {
                    println!("client disconnected unexpectedly");
                    completed.push(i); // 👈
                    continue 'next;
                }
                Ok(n) => *read += n,
                // 当前连接未准备好，先处理下一个活动连接
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue 'next,
                Err(e) => panic!("{e}"),
            }

            // ...
        }

        // ...
    }
}

// 按相反顺序迭代以保留索引
for i in completed.into_iter().rev() {
    connections.remove(i); // 👈
}

```
读操作完成后，我们必须切换到 Write 状态并尝试写入回应。写操作的逻辑跟读操作非常相似，写操作完成后，需要切换到 Flush 状态。
```rust
if let ConnectionState::Read { request, read } = state {
    // ...

    // 切换到写状态
    let response = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Length: 12\n",
        "Connection: close\r\n\r\n",
        "Hello world!"
    );

    *state = ConnectionState::Write { // 👈
        response: response.as_bytes(),
        written: 0,
    };
}

if let ConnectionState::Write { response, written } = state {
    loop {
        match connection.write(&response[*written..]) {
            Ok(0) => {
                println!("client disconnected unexpectedly");
                completed.push(i);
                continue 'next;
            }
            Ok(n) => {
                *written += n;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // 当前连接的操作还未就绪，继续处理下一个连接
                continue 'next;
            }
            Err(e) => panic!("{e}"),
        }

        // 判断响应数据是否已全部写入完毕
        if *written == response.len() {
            break;
        }
    }

    // 写操作完成，进入 Flush 状态
    *state = ConnectionState::Flush;
}

```
成功完成刷新操作后，我们标记当前连接为完成，并从 completed 列表中移除。
```rustif let ConnectionState::Flush = state {
    match connection.flush() {
        Ok(_) => {
            completed.push(i); // 👈
        },
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            // 当前连接的操作还未就绪，继续处理下一个连接
            continue 'next;
        }
        Err(e) => panic!("{e}"),
    }
}

```
就是这样！以下是新的更高水平的 web 服务流程：
```rust
fn main() {
    // 绑定侦听器
    let listener = TcpListener::bind("localhost:3000").unwrap();
    listener.set_nonblocking(true).unwrap();

    let mut connections = Vec::new();

    loop {
        // 尝试接受一个连接请求
        match listener.accept() {
            Ok((connection, _)) => {
                connection.set_nonblocking(true).unwrap();

                // 跟踪连接状态
                let state = ConnectionState::Read {
                    request: Vec::new(),
                    read: 0,
                };

                connections.push((connection, state));
            },
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("{e}"),
        }

        let mut completed = Vec::new();

        // 尝试驱动活动连接向前进展
        'next: for (i, (connection, state)) in connections.iter_mut().enumerate() {
            if let ConnectionState::Read { request, read } = state {
                // ...
                *state = ConnectionState::Write { response, written };
            }

            if let ConnectionState::Write { response, written } = state {
                // ...
                *state = ConnectionState::Flush;
            }

            if let ConnectionState::Flush = state {
                // ...
            }
        }

        // 保持索引不变，反序遍历 completed 列表，删除已完成操作的连接
        for i in completed.into_iter().rev() {
            connections.remove(i);
        }
    }
}

```
现在，我们必须自己管理调度，事情变得越来越复杂了……

关键的时候来了……
```
$ curl localhost:3000
```
工作正常！
