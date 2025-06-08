```rust
use alloc::sync::Arc;

/// 唤醒特定任务的一种方式。
///
/// 实现该 trait 的类型，通常会被包装在 `Arc` 中，
/// 并可被转换为 [`Waker`] 对象。
/// 这些 Waker 可用于通知执行器某个任务已准备好再次被 `poll`。
///
/// 目前有两种方式可以将实现了 `ArcWake` 的类型转换为 [`Waker`]：
///
/// - [`waker`](super::waker())：将 `Arc<impl ArcWake>` 转换为 [`Waker`]；
/// - [`waker_ref`](super::waker_ref())：将 `&Arc<impl ArcWake>` 转换为 [`WakerRef`]，
///   它提供对 [`&Waker`] 的访问。
///
/// [`Waker`]: std::task::Waker  
/// [`WakerRef`]: super::WakerRef
///
/// **注意：**此 trait 要求类型满足 `Send + Sync`，
/// 因为 `Arc<T>` 并不会自动推导出这两个 trait，
/// 而 [`Waker`] 本身实现了 `Send + Sync`。
pub trait ArcWake: Send + Sync {
    /// 表示关联的任务已经可以继续执行，应当被 `poll`。
    ///
    /// 这个函数可以从任意线程调用，包括并非创建该 `ArcWake` 的 [`Waker`] 的线程。
    ///
    /// 执行器通常维护一个“就绪任务队列”，`wake` 应该将对应任务加入这个队列中。
    ///
    /// [`Waker`]: std::task::Waker
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self)
    }

    /// 表示关联的任务已经可以继续执行，应当被 `poll`。
    ///
    /// 这个函数可以从任意线程调用，包括并非创建该 `ArcWake` 的 [`Waker`] 的线程。
    ///
    /// 执行器通常维护一个“就绪任务队列”，`wake_by_ref` 应该将对应任务加入这个队列中。
    ///
    /// 该函数与 [`wake`](ArcWake::wake) 类似，但它不能消费传入的数据指针（即不会移动或销毁 Arc 实例）。
    ///
    /// [`Waker`]: std::task::Waker
    fn wake_by_ref(arc_self: &Arc<Self>);
}


```
- 实现该 trait 的类型，通常会被包装在 `Arc` 中,并可被转换为` [`Waker`] `对象。
	- 转换机制
		-  `futures::task：waker`：将 `Arc<impl ArcWake>` 转换为 `[`Waker`]`；
		- `futures::task：waker_ref`：将 `&Arc<impl ArcWake>` 转换为 `[`WakerRef`]`，它提供对` [`&Waker`]` 的访问。
- 作用：这些 Waker 可用于通知执行器某个任务已准备好再次被 `poll`。
