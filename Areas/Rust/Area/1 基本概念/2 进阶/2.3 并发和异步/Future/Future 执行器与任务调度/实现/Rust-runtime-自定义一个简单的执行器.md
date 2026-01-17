
## 1. 核心观点  



## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

- 看下Future的函数的poll（）函数[1.0.1 RustBook Future](../Future/1.0.1%20RustBook%20Future.md)
- waker：
	- [Rust-future-waker-基本概念](../../wake/Rust-future-waker-基本概念.md)
- 核心问题：
	- 如何实现这个waker，调用waker=>再次poll的效果
	- 可以使用消息通道
		- 发送者，负责创建任务，任务中包含Future，发送任务到通道
		- 接受者，接受任务，执行任务中Future的Poll方法
		- 唤醒机制实现，waker：就是再次将任务发送到消息通道
- 核心设计
	- 我们的执行者通过发送任务在通道上运行来工作。执行人会从通道中提取事件并运行。当任务准备继续工作（被唤醒）时，它可以通过重新进入通道来安排再次轮询。
	- 在这种设计中，执行者本身只需要任务通道的接收端。用户会获得发送端，以便生成新的未来。任务本身就是可以重新调度的future，所以我们会将它们存储为future，并与任务可以用来重新排队的发送者一起存储。


### 实际实现
#### 任务结构体
- [构建Waker](../构建Waker.md)有两种方式，我们使用高级抽象[ArcWaker](../ArcWaker.md)
- 任务封装future对象，直接在任务对象上实现wake机制
- 任务结构
	- 一个future
	- 一个发送者对象
结构体如下：
- 因为ArcWaker要求
	- `pub trait ArcWake: Send + Sync`
- `Mutex` 的来包装future，**主要是为了满足唤醒机制所需的类型约束和线程安全性要求**，这并非业务逻辑的必需，而是 **由 Rust 的异步模型 + 类型系统 + `Waker` 机制共同决定的**。
- 'static:
	- [操作系统-线程-线程栈之间互相访问](../../../../../../../../basic/操作系统/os-note/note/atomic/操作系统-线程-线程栈之间互相访问.md)不用’static也可以使用send但是容易不安全
```rust
  
/// 一个Future，它可以调度自己(将自己放入任务通道中)，然后等待执行器去`poll`

struct Task {

    /// 进行中的Future，在未来的某个时间点会被完成

    ///

    /// 按理来说`Mutex`在这里是多余的，因为我们只有一个线程来执行任务。但是由于

    /// Rust并不聪明，它无法知道`Future`只会在一个线程内被修改，并不会被跨线程修改。因此

    /// 我们需要使用`Mutex`来满足这个笨笨的编译器对线程安全的执着。

    ///

    /// 如果是生产级的执行器实现，不会使用`Mutex`，因为会带来性能上的开销，取而代之的是使用`UnsafeCell`

    future: Mutex<Option<BoxFuture<'static, ()>>>,

  

    /// 可以将该任务自身放回到任务通道中，等待执行器的poll

    task_sender: SyncSender<Arc<Task>>,

}
```
#### 实现ArcWake的wake_by_ref，实现唤醒逻辑
```rust
impl ArcWake for Task {

    fn wake_by_ref(arc_self: &Arc<Self>) {

        // 通过发送任务到任务管道的方式来实现`wake`，这样`wake`后，任务就能被执行器`poll`

        let cloned = arc_self.clone();

        arc_self

            .task_sender

            .send(cloned)

            .expect("任务队列已满");

    }

}
```

- 如何标记该任务需要重新被调度，使用队列重新发送

#### 实现初始任务逻辑
```rust
/// `Spawner`负责创建新的`Future`然后将它发送到任务通道中

#[derive(Clone)]

struct Spawner {

    task_sender: SyncSender<Arc<Task>>,

}
```
#### 实现初始任务逻辑
```rust
impl Spawner {

    fn spawn(&self, future: impl Future<Output = ()> + 'static + Send) {

        let future = future.boxed();

        let task = Arc::new(Task {

            future: Mutex::new(Some(future)),

            task_sender: self.task_sender.clone(),

        });

        self.task_sender.send(task).expect("任务队列已满");

    }

}
```
 1. `+ 'static`：意味着 **`Future` 中不包含非 `'static` 的引用**
	 - `static` 意味着：生命周期不依赖于外部栈帧
		 - 要么是堆上的所有权数据（例如 `String`, `Vec`, `Box<_>`）
		 - 要么是 `'static` 变量
	 - `+ Send`：意味着 **`Future` 可以安全地在不同线程之间移动**
		 - 因为 `Waker` 是可以被**跨线程唤醒的**！

- 创建一个新任务
	- 任务拥有发送端
	- 任务拥有可以跨线程的future
- 发送任务
- 方法结束后原来的发送者只会存在一个所有权对象，就是任务中的所有权对象
### 实现执行者逻辑
```rust
/// 任务执行器，负责从通道中接收任务然后执行

struct Executor {

    ready_queue: Receiver<Arc<Task>>,

}
```

```rust
impl Executor {

    fn run(&self) {

        while let Ok(task) = self.ready_queue.recv() {

            // 获取一个future，若它还没有完成(仍然是Some，不是None)，则对它进行一次poll并尝试完成它

            let mut future_slot = task.future.lock().unwrap();

            if let Some(mut future) = future_slot.take() {

                // 基于任务自身创建一个 `LocalWaker`

                let waker = waker_ref(&task);

                let context = &mut Context::from_waker(&*waker);

                // `BoxFuture<T>`是`Pin<Box<dyn Future<Output = T> + Send + 'static>>`的类型别名

                // 通过调用`as_mut`方法，可以将上面的类型转换成`Pin<&mut dyn Future + Send + 'static>`

                if future.as_mut().poll(context).is_pending() {

                    // Future还没执行完，因此将它放回任务中，等待下次被poll

                    *future_slot = Some(future);

                }

            }

        }

    }

}
```

- 发送器行为
	- **如果队列非空**，立刻取出一个值；
	- **如果队列为空**，**但发送端（Sender）还存在**，线程会 **阻塞等待新任务**；
	- **如果所有 Sender 都被 drop 掉了**，那么 `recv()` 会返回 `Err(RecvError)`，从而跳出 `while` 循环，执行器退出。
- 队列非空，取出任务，**基于任务创建一个waker，调用poll方法，任务没执行完将任务放回**
- 将该任务的wake逻辑通过上下文传递给future的poll方法，poll方法就可以接受该waker然后传递给应该执行唤醒的外部事物
### main
```rust
fn main() {

    let (executor, spawner) = new_executor_and_spawner();

  

    // 生成一个任务

    spawner.spawn(async {

        println!("howdy!");

        // 创建定时器Future，并等待它完成

        TimerFuture::new(Duration::new(2, 0)).await;

        println!("done!");

    });

    // drop掉发送器，任务存着发送器，任务没了，发送器没了，执行器就会退出

    drop(spawner);

  

    // 运行执行器直到任务队列为空

    // 任务运行后，会先打印`howdy!`, 暂停2秒，接着打印 `done!`

    executor.run();

}
```
#### new_executor_and_spawner()
```rust
fn new_executor_and_spawner() -> (Executor, Spawner) {

    // 任务通道允许的最大缓冲数(任务队列的最大长度)

    // 当前的实现仅仅是为了简单，在实际的执行中，并不会这么使用

    const MAX_QUEUED_TASKS: usize = 10_000;

    let (task_sender, ready_queue) = sync_channel(MAX_QUEUED_TASKS);

    (Executor { ready_queue }, Spawner { task_sender })

}

```
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  


