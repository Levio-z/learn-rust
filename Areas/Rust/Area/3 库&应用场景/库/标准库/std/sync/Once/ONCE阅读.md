### 数据状态
```rust
// On some platforms, the OS is very nice and handles the waiter queue for us.
//  在某些平台上，操作系统非常贴心地帮我们处理了等待线程队列。

// This means we only need one atomic value with 4 states:
//  这意味着我们只需要一个原子值来表示 4 种状态：
/// No initialization has run yet, and no thread is currently using the Once.
/// 尚未进行初始化，也没有线程正在使用该 Once。
const INCOMPLETE: Primitive = 0;

/// Some thread has previously attempted to initialize the Once, but it panicked,
/// so the Once is now poisoned. There are no other threads currently accessing
/// this Once.
/// 有线程曾尝试初始化 Once，但发生了 panic，因此该 Once 现在处于“中毒”状态。
/// 当前没有其他线程访问该 Once。
const POISONED: Primitive = 1;

/// Some thread is currently attempting to run initialization. It may succeed,
/// so all future threads need to wait for it to finish.
/// 有线程当前正在尝试执行初始化操作。它可能会成功，
/// 所以所有后续线程都需要等待其完成。
const RUNNING: Primitive = 2;


/// Initialization has completed and all future calls should finish immediately.
/// 初始化已完成，所有后续调用应立即返回。
const COMPLETE: Primitive = 3;

// An additional bit indicates whether there are waiting threads:
//  还有一个附加的位用于表示是否有等待的线程：
/// May only be set if the state is not COMPLETE.
/// 仅当状态不是 COMPLETE 时，才可以设置此位。
const QUEUED: Primitive = 4;
// Threads wait by setting the QUEUED bit and calling `futex_wait` on the state
// variable. When the running thread finishes, it will wake all waiting threads using
// `futex_wake_all`.
//
// 线程通过设置 QUEUED 位并调用 `futex_wait` 来等待该状态变量；
// 当前正在初始化的线程在完成时，会使用 `futex_wake_all` 唤醒所有等待线程。

const STATE_MASK: Primitive = 0b11;
// 用于提取低两位的状态字段（即 INCOMPLETE/POISONED/RUNNING/COMPLETE）

```



### call_one
```rust
    pub fn call_once<F>(&self, f: F)

    where

        F: FnOnce(),

    {

        // Fast path check

        if self.inner.is_completed() {

            return;

        }

  

        let mut f = Some(f);

        self.inner.call(false, &mut |_| f.take().unwrap()());

    }
```
#### Fast path check

```
#[inline]

    pub fn is_completed(&self) -> bool {

        // Use acquire ordering to make all initialization changes visible to the

        // current thread.

        self.state_and_queued.load(Acquire) == COMPLETE

    }
```

#### call
```rust
#[cold]

    #[track_caller]

    pub fn call(&self, ignore_poisoning: bool, f: &mut dyn FnMut(&public::OnceState)) {

        let mut state_and_queued = self.state_and_queued.load(Acquire);

        loop {

            let state = state_and_queued & STATE_MASK;

            let queued = state_and_queued & QUEUED != 0;

            match state {

                COMPLETE => return,

                POISONED if !ignore_poisoning => {

                    // Panic to propagate the poison.

                    panic!("Once instance has previously been poisoned");

                }

                INCOMPLETE | POISONED => {

                    // Try to register the current thread as the one running.

                    let next = RUNNING + if queued { QUEUED } else { 0 };

                    if let Err(new) = self.state_and_queued.compare_exchange_weak(

                        state_and_queued,

                        next,

                        Acquire,

                        Acquire,

                    ) {

                        state_and_queued = new;

                        continue;

                    }

  

                    // `waiter_queue` will manage other waiting threads, and

                    // wake them up on drop.

                    let mut waiter_queue = CompletionGuard {

                        state_and_queued: &self.state_and_queued,

                        set_state_on_drop_to: POISONED,

                    };

                    // Run the function, letting it know if we're poisoned or not.

                    let f_state = public::OnceState {

                        inner: OnceState {

                            poisoned: state == POISONED,

                            set_state_to: Cell::new(COMPLETE),

                        },

                    };

                    f(&f_state);

                    waiter_queue.set_state_on_drop_to = f_state.inner.set_state_to.get();

                    return;

                }

                _ => {

                    // All other values must be RUNNING.

                    assert!(state == RUNNING);

  

                    // Set the QUEUED bit if it is not already set.

                    if !queued {

                        state_and_queued += QUEUED;

                        if let Err(new) = self.state_and_queued.compare_exchange_weak(

                            state,

                            state_and_queued,

                            Relaxed,

                            Acquire,

                        ) {

                            state_and_queued = new;

                            continue;

                        }

                    }

  

                    futex_wait(&self.state_and_queued, state_and_queued, None);

                    state_and_queued = self.state_and_queued.load(Acquire);

                }

            }

        }

    }
```
##### INCOMPLETE | POISONED
```rust
if let Err(new) = self.state_and_queued.compare_exchange_weak(
    state_and_queued,
    next,
    Acquire,
    Acquire,
) {
    state_and_queued = new;
    continue;
}


```
无论成功还是失败，都是用 `Acquire` 作为内存序（Ordering）
成功时用 Acquire：
- 这时当前线程成功把状态更新为 `RUNNING`（或者 `RUNNING | QUEUED`），代表线程获得了执行初始化的“权利”。
	- 典型的同步语义：先读取状态，然后安全地访问初始化数据。
	- 使用 `Acquire` 保证后续的读写操作不会乱序到 CAS 之前，保证线程看到初始化过程中之前写入的数据。
失败时也用 Acquire：
- CAS 失败意味着当前值已被其他线程修改，当前线程读到了一个新的值。
- 读取这个新值必须同步地看到其他线程之前写入的数据，否则可能看到“过时”或者“不一致”的状态。
- 因此失败路径也是一次**读取操作**，需要使用 `Acquire` 保证从内存中读取最新的同步状态。
- 这使得循环条件能正确观察状态变化，避免因为乱序导致读取到陈旧状态，造成死循环或逻辑错误。
```rust
// `waiter_queue` will manage other waiting threads, and
// wake them up on drop.
let mut waiter_queue = CompletionGua
    state_and_queued: &self.state_and_queued,
    set_state_on_drop_to: POISONED,
};
```
它是一个 **RAII 风格（Resource Acquisition Is Initialization）** 的“看门人”对象，用于确保在 `Once` 初始化结束时，无论函数是否成功执行，都能**正确地更新状态并唤醒等待线程**。下面详细解释其含义、目的、原理与使用场景。
- **作用**：一旦当前线程（初始化线程）完成初始化逻辑，即使函数中间 panic 或 return，该 guard 在销毁（drop）时都会自动更新状态并唤醒其他等待线程。
- **默认行为**：初始化线程还未完成之前，先设定默认“失败状态”：`POISONED`。
- **若成功初始化**：稍后会根据用户代码执行结果，将其状态变为 `COMPLETE` 或其它。
>“我准备跑初始化代码了。如果我中途挂了，就把状态标成中毒（POISONED）；否则我会在成功后手动修改这个值。”
```rust
impl<'a> Drop for CompletionGuard<'a> {

    fn drop(&mut self) {

        // Use release ordering to propagate changes to all threads checking

        // up on the Once. `futex_wake_all` does its own synchronization, hence

        // we do not need `AcqRel`.

        if self.state_and_queued.swap(self.set_state_on_drop_to, Release) & QUEUED != 0 {

            futex_wake_all(self.state_and_queued);

        }

    }

}
```
- 使用 `Release` ordering 的目的是
	- 保证 **初始化线程在执行完用户函数 `f()` 后写入的所有内存（data）变动**，
	- 会在 `state_and_queued` 被设置为 `COMPLETE`（或 `POISONED`）之前 **先行发生**；
- `futex_wake_all` 自带同步语义，确保唤醒的线程可以立即读取内存。
- 这里的 `swap` 指的是 **原子交换操作**，是原子类型（如 `AtomicU32`）提供的一种操作方法：
	- **将当前原子变量的值替换为 `val`**（这里是 `self.set_state_on_drop_to`，即 `COMPLETE` 或 `POISONED`）；
	- **返回替换前的旧值**；
	- 整个操作是 **原子的（atomic）** —— 不可中断地完成。


Rust 的所有权模型确保：
- 当 `waiter_queue` 变量离开作用域或函数返回时，其 `Drop` 实现会自动执行。
- 在 `Drop` 中会检查 `set_state_on_drop_to` 并将 `state_and_queued` 设置为对应值，同时唤醒所有等待线程（例如使用 futex）。
```
1. 当前线程进入 Once 初始化代码块
2. 创建 CompletionGuard，状态默认是 POISONED
3. 执行用户提供的初始化逻辑 f(&f_state)
4. f 逻辑执行成功后，设置 set_state_on_drop_to = COMPLETE
5. 离开作用域时自动 drop
   ⤷ 触发 CompletionGuard::drop:
       - 根据 set_state_on_drop_to 设置最终状态
       - futex_wake_all 叫醒其他线程

```
该 RAII guard 设计适用于：

| 场景       | 原因                         |
| -------- | -------------------------- |
| 多线程并发初始化 | 避免初始化竞争和竞态条件               |
| 保证状态始终收尾 | 无论 panic、错误或正常退出都能统一释放     |
| 安全唤醒等待线程 | 防止线程永久挂起或死锁                |
| 封装状态转换逻辑 | 将一次性状态管理抽象为 Drop 过程，减少出错机会 |
```rust
// Run the function, letting it know if we're poisoned or not.
let f_state = public::OnceState {
    inner: OnceState {
        poisoned: state == POISONED,
        set_state_to: Cell::new(COMPLETE),
    },
};
f(&f_state);
waiter_queue.set_state_on_drop_to = f_state.inner.set_state_to.get();
return;
```
> 运行一次性初始化函数 `f(&OnceState)`，并记录初始化是否成功；成功则状态设为 `COMPLETE`，失败则设为 `POISONED`。

构造传入用户函数的状态对象 `OnceState`
```rust
let f_state = public::OnceState {
    inner: OnceState {
        poisoned: state == POISONED,
        set_state_to: Cell::new(COMPLETE),
    },
};
```

- `public::OnceState` 是暴露给用户的接口包装器（它持有内部的 `OnceState` 实例）。
- `poisoned: state == POISONED` 表示 **如果先前某线程在初始化时 panic，则当前为中毒状态**。
- `set_state_to: Cell::new(COMPLETE)` 表示 **默认执行成功后将状态标记为 `COMPLETE`**。但注意：这个值是可变的！
```
f(&f_state);
```
执行用户提供的初始化函数
用户可通过传入的 `OnceState`：
- 查询当前是否处于中毒状态
- 决定是否要恢复或终止初始化
- 甚至可以调用 `set_poisoned()` 主动让当前初始化标记为失败（写入 `POISONED`）


根据初始化结果设置最终状态
```rust
waiter_queue.set_state_on_drop_to = f_state.inner.set_state_to.get();
```


```rust
_ => {

                    // All other values must be RUNNING.

                    assert!(state == RUNNING);

  

                    // Set the QUEUED bit if it is not already set.

                    if !queued {

                        state_and_queued += QUEUED;

                        if let Err(new) = self.state_and_queued.compare_exchange_weak(

                            state,

                            state_and_queued,

                            Relaxed,

                            Acquire,

                        ) {

                            state_and_queued = new;

                            continue;

                        }

                    }

  

                    futex_wait(&self.state_and_queued, state_and_queued, None);

                    state_and_queued = self.state_and_queued.load(Acquire);

                }

```
这里 `_` 表示除了 `INCOMPLETE` / `POISONED` 情况外的其他状态，唯一合法值就是 `RUNNING`。因此断言 `state == RUNNING` 是一种防御性编程手段，防止状态机被破坏。
加入标记队列
```rust
if !queued {
    state_and_queued += QUEUED;
    if let Err(new) = self.state_and_queued.compare_exchange_weak(
        state,
        state_and_queued,
        Relaxed,
        Acquire,
    ) {
        state_and_queued = new;
        continue;
    }
}

```
- 如果还没设置 `QUEUED` 标志（表示当前线程还没“排队”），就加上它。
- 使用 CAS（比较交换）尝试将状态从 `RUNNING` → `RUNNING | QUEUED`。
为什么要设置 `QUEUED`？
用于告诉运行初始化的线程："有其他线程在等待你完成初始化"。这样等它初始化完后，就能唤醒这些等待线程。
为什么使用 `Relaxed/Acquire` 语义？
- 成功路径（`Relaxed`）：我们只是设置一个 _位标志_，不涉及数据同步。
- 失败路径（`Acquire`）：需要确保如果状态发生变化（如 COMPLETE），能 **看到对共享数据的写入**（跨线程同步）。

等待初始化完成
```rust
   futex_wait(&self.state_and_queued, state_and_queued, None);
```
当前线程阻塞，直到初始化线程将状态设置为 `COMPLETE` 或 `POISONED`。
这个 `futex_wait` 类似于：
- Linux 上的 `futex()`；
- 等待某个原子变量的值发生变化；
- 在用户态快速轮询，在必要时陷入内核态睡眠，提高效率。
```rust
pub fn futex_wait<W: Waitable>(futex: &W::Futex, expected: W, timeout: Option<Duration>) -> bool {

    // return false only on timeout

    wait_on_address(futex, expected, timeout) || api::get_last_error() != WinError::TIMEOUT

}
```
- 是 Rust 在 **Windows 平台上的 Futex（Fast Userspace Mutex）仿真实现**之一，用于在用户态高效地等待一个原子变量达到某个值，或者在一段时间内发生变化。
	- `W: Waitable`: 一个 trait，代表某种可以“等待”的原子类型，比如 `AtomicU32`、`AtomicI32` 等。
	- `W::Futex`: 实际要等待的地址类型（通常是 `*const i32` 等裸指针）。
	- `expected: W`: 期望值，仅当当前值等于它时才进入等待（防止竞争丢失）。
	- `timeout: Option<Duration>`: 最长等待时间。`None` 表示无限等待。
- 返回值：
	- `bool`: 是否被**唤醒**。
	    - `true`：被其他线程显式唤醒（如 `futex_wake`）或值变化。
	    - `false`：发生**超时**（timeout）。
-  `wait_on_address(...)`
	- 封装了 Windows 的 [`WaitOnAddress`](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitonaddress) API。
	- 会检查 `*futex` 当前值是否等于 `expected`，**只有匹配时才等待**。
	- 如果等到值变化或被唤醒，则返回 `true`。
	- 如果超时或错误，返回 `false`。
	- 只有当共享内存地址中的当前值 == 期望值 expected 时，WaitOnAddress 才会挂起当前线程。
		否则，它会立即返回，不阻塞。
	- 这就是经典的 **抢锁失败 -> 判断值没变 -> 才等** 的流程。
- api::get_last_error() != WinError::TIMEOUT
	-  `wait_on_address` 返回 `false` 时，我们不能立即判断就是“超时”。
	- 有可能是被唤醒了，但系统发生了非超时的错误。
	- 所以我们调用 `GetLastError()` 检查是否是 `TIMEOUT` 错误码。
	- 如果 **不是超时错误**，则推断它被唤醒，返回 `true`。
```rust
_and_queued = self.state_and_queued.load(Acquire);
```
是用来 **重新加载最新的原子状态值**，并且通过 `Acquire` 内存顺序 **同步其他线程已写入的共享内存内容**。下面我从语义、目的、为何使用 `Acquire`、配合场景四个维度为你详细解释：
从原子变量 `self.state_and_queued` 中 **读取当前值（状态）**，并赋值给 `state_and_queued` 本地变量，作为接下来的控制依据。
它的作用是：
1. 等待期间（可能几毫秒或更长），有**其他线程修改了该原子值**
2. 当前线程从睡眠中唤醒后，**必须重新加载一次最新状态**来判断下一步行为（如是否继续等待 / 退出）
3. 避免使用旧的 `state_and_queued` 值导致逻辑错误（例如误判状态）