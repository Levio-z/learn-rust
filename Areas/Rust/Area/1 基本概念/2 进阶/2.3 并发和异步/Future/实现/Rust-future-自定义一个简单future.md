---
tags:
  - note
---
## 1. 核心观点  

- [Rust-future-waker-基本概念](../wake/Rust-future-waker-基本概念.md)


- 模拟一个Future，数据就准备好，会**调用wake()通知执行器**
- 这里就是在另一个线程中，睡眠指定时间模拟数据在一段时间准备好的情况
- 需要一个共享数据
	- **共享一个条件**：
		- 满足，返回
		- 不满足，注册一个waker到外部
			- **共享一个waker**：future poll的时候，可以把这个waker访问权限给外部，外面数据准备就绪，**修改条件，并且可以调用wake唤醒这个程序**让他返回完成


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


### 实现
#### 共享数据结构
```rust
struct SharedState {

    complete: bool,

    // 当睡眠结束后，线程可以用waker通知TimerFuture来唤醒任务

    waker: Option<Waker>,

}
```
#### 实现future
```rust
pub struct TimerFuture {

    // 用于在新线程和 Future 定时器间共享。

    shared_state: Arc<Mutex<SharedState>>,

}
```
Future Poll的核心逻辑，根据条件返回不同任务状态，值准备好了还是pending，pending需要设置好waker，让外部唤醒执行器再次执行poll
```rust
impl Future for TimerFuture {

    type Output = ();

    // 可以被任务调度器（比如 Future 调度器）移动 Pin 自身，但不会移动 T 的内容。

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {

        // 通过检查共享状态，来确定定时器是否完成

        let mut shared_state = self.shared_state.lock().unwrap();

        if shared_state.complete {

            Poll::Ready(())

        } else {

            // 设置waker，新线程在睡眠计时结束时，唤醒当前任务，接着再次对Future进行Poll操作

            // 下面的clone每次被`poll`都会发生一次，实际上，应该是只`clone`一次更加合理

            // 每次`cloen`的原因是：`TimerFuture` 可以在执行器的不同任务间移动，如果只克隆一次

            // 那么每次获取到的`waker`可能已经被篡改指向其他任务，导致执行器运行了错误的任务

            shared_state.waker = Some(cx.waker().clone());

            Poll::Pending

        }

    }

}
```
外部数据准备好了，唤醒任务
```rust
impl TimerFuture {

    /// 创建一个新的`TimerFuture`，在指定的时间结束后，该`Future`可以完成

    pub fn new(duration: Duration) -> Self {

        let shared_state = Arc::new(Mutex::new(SharedState {

            complete: false,

            waker: None,

        }));

  

        // 创建新线程

        let thread_shared_state = shared_state.clone();

        thread::spawn(move || {

            // 睡眠指定时间实现计时功能

            thread::sleep(duration);

            let mut shared_state = thread_shared_state.lock().unwrap();

            // 通知执行器定时器已经完成，可以继续`poll`对应的`Future`了

            shared_state.complete = true;

            if let Some(waker) = shared_state.waker.take() {

                waker.wake()

            }

        });

  

        TimerFuture { shared_state }

    }

}
```
## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-future-waker-基本概念](../wake/Rust-future-waker-基本概念.md)
- 后续卡片：
	- [Rust-future-组合设计的思想-基本概念](../组合子/Rust-future-组合设计的思想-基本概念.md)
	- [Rust-future-状态机](../Rust-future-状态机.md)
	- [Rust-Pinning-基本概念](../../Pin/Rust-Pinning-基本概念.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
