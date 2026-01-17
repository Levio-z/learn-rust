---
tags:
  - note
---

## 1. 核心观点  



## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

例如，考虑我们想从一个可能已有也可能没有数据的套接字读取数据的情况。如果有数据，我们可以读取并返回 `Poll：：Ready（data）`;但如果没有数据准备好，我们的未来就会被阻挡，无法继续前进。当没有可用数据时，我们必须注册`唤醒` ，以便在数据准备好时调用，这将告诉执行者我们的未来已准备好推进。一个简单的 `SocketRead` 未来可能如下：
```rust
pub struct SocketRead<'a> {
    socket: &'a Socket,
}

impl SimpleFuture for SocketRead<'_> {
    type Output = Vec<u8>;

    fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
        if self.socket.has_data_to_read() {
            // The socket has data -- read it into a buffer and return it.
            Poll::Ready(self.socket.read_buf())
        } else {
            // The socket does not yet have data.
            //
            // Arrange for `wake` to be called once data is available.
            // When data becomes available, `wake` will be called, and the
            // user of this `Future` will know to call `poll` again and
            // receive data.
            self.socket.set_readable_callback(wake);
            Poll::Pending
        }
    }
}
```



在现实中，像网络服务器这样的复杂应用可能有成千上万个不同的连接，这些连接的唤醒都应分别管理。`Context` 类型通过提供访问类型`为 Waker` 的值来解决这个问题，该值可用于唤醒特定任务。

[Rust-future-源码-waker](Rust-future-源码-waker.md)

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  
