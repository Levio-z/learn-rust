好吧，我们的服务器现在能在单一线程中同时处理多个请求了。感谢 epoll，它在我们的工作场景中非常高效。但是，仍然还存在问题。

我们需要自己规划任务的执行，需要自己考虑如何高效地调度任务，这使得我们的代码复杂度急剧增加。

任务的执行，也从一个简单的顺序执行循环变成了庞大的事件循环，需要管理多个状态机。

总感觉差点意思。

使我们的原始服务器成多线程非常简单，只需在 thread::spawn 中添加一行代码即可。仔细想想，我们的服务器仍然是一组并发任务，只是我们在一个巨大的循环中混乱地管理它们。

这让扩展功能变得非常困难，在程序中添加的功能越多，循环就变得越复杂，因为所有东西都紧密地耦合在一起。如果可以编写一个类似 thread::spawn 的抽象，能让我们将任务写成独立的单元，能集中在一个地方处理所有任务的调度和事件处理，从而重新获得流程控制权，会怎么样呢？

好吧，我们的服务器现在能在单一线程中同时处理多个请求了。感谢 epoll，它在我们的工作场景中非常高效。但是，仍然还存在问题。

我们需要自己规划任务的执行，需要自己考虑如何高效地调度任务，这使得我们的代码复杂度急剧增加。

任务的执行，也从一个简单的顺序执行循环变成了庞大的事件循环，需要管理多个状态机。

总感觉差点意思。

使我们的原始服务器成多线程非常简单，只需在 thread::spawn 中添加一行代码即可。仔细想想，我们的服务器仍然是一组并发任务，只是我们在一个巨大的循环中混乱地管理它们。

> 这种思想被称为异步编程。

我们来看看 thread::spawn 的函数签名：
```rust
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static;

```
thread::spawn 接受一个闭包，但我们的版本其实并不能，因为我们不是操作系统，不能随意抢占代码。我们需要**以某种方式来表达一项不受阻碍、可恢复的任务。**
```rust
// fn spawn<T: Task>(task: T);

trait Task {}

```
处理一个请求是一个任务,从连接读取或写入数据也是。一个任务实质上是一段待执行的代码，代表着它将在未来某个时候需要得到执行。

Future (译注：本意是未来，在异步中我们保留原文代表这种 trait)，确实是个好名字，不是吗？
```rust
trait Future {
    type Output;

    fn run(self) -> Self::Output;
}

```
这个签名并不能工作。run 函数直接返回 Self::Output 意味着它会阻塞直到返回，而这正是我们极力在避免的。我们要寻找其他方法，实现不阻塞的同时，能推动我们的 future 前进，就像我们之前在事件循环的状态机中实现的那样。

实际上，在执行一个 future 的时候，我们要做的就是询问它是否已就绪，_轮询( polling )_ 它，然后给它执行的机会。
```rust
trait Future {
    type Output;

    fn poll(self) -> Option<Self::Output>;
}

```
看起来差不多了。

但是，如果我们多次调用 poll，除了等着，我们并不能获取 self， 所以它应该是一个引用，一个可变的引用，通过它，我们可以改变任务内部的状态，比如 ConnectState。
```rust
trait Future {
    type Output;

    fn poll(&mut self) -> Option<Self::Output>;
}

```
现在，来设想一下执行这些 future 的调度器：
```rust
impl Scheduler {
    fn run(&self) {
        loop {
            for future in &self.tasks {
                future.poll();
            }
        }
    }
}
```
这看起来不怎么样。
future 初始化完成后，当 epoll 返回的它的事件时，调度器调用它的 poll 方法来给它一个执行的机会。

如果 future 是 I/O 操作，在 接到 epoll 通知时我们就知道它可以执行了。问题是我们不知道 epoll 事件对应的是哪个 future, 因为 future 的执行过程都在内部的 poll 中。

**调度器需要传递一个 ID 给 future，它可以用这个 ID 而不是文件描述符向 epoll 注册任何 I/O 资源**。通过这种方式，调度器就能把 epoll 事件和 future 对应起来了。

```rust
impl Scheduler {
    fn spawn<T>(&self, mut future: T) {
        let id = rand();
        // 对 future 调用一次轮询让它运转起来，传入的参数是它的 ID
        future.poll(event.id);
        // 保存 future
        self.tasks.insert(id, future);
    }

    fn run(self) {
        // ...

        for event in epoll_events {
            // 根据事件 ID 轮询相应的 future
            let future = self.tasks.get(&event.id).unwrap();
            future.poll(event.id);
        }
    }
}

```
您知道的，**如果有一种更通用的方式来告诉调度器 future 当前的进度，而不是把每个 future 都绑定到 epoll，那就太好了**。future 有不同的类型，每种类型都可能有不同的执行方式，比如在后台线程中执行的定时器、或者是一个通道，它需要在消息已就绪的时候通知相应的任务。
如果我们给 future 更多的控制权呢？ 如果我们不是简单地用一个 ID, 而是给每个 future 一个能唤醒自己的方法，能通知调度器它已经准备好可以执行了呢？
一个简单的回调函数就可以做到。
```rust
#[derive(Clone)]
struct Waker(Arc<dyn Fn() + Send + Sync>);

impl Waker {
    fn wake(&self) {
        (self.0)()
    }
}

trait Future {
    type Output;

    fn poll(&mut self, waker: Waker) -> Option<Self::Output>;
}

```
**调度器可以为每个 future 提供一个回调函数，它被调用时更新该 future 在调度器中的状态，标记 future 为就绪。这样调度器就完全和 epoll 或其他任何独立通知系统解耦了。**

唤醒器 ( Waker ) 是线程安全的，允许我们使用后台线程唤醒 future。目前所有的任务都已连接到 epoll，这马上就会派上用场了。

