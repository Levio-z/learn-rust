https://rustwiki.org/zh-CN/std/thread/fn.spawn.html
```rust
pub fn spawn<F, T>(f: F) -> JoinHandle<T>where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
```



产生一个新线程，并为其返回一个 [`JoinHandle`](https://rustwiki.org/zh-CN/std/thread/struct.JoinHandle.html "struct std::thread::JoinHandle")。

连接句柄提供了一个 [`join`](https://rustwiki.org/zh-CN/std/thread/struct.JoinHandle.html#method.join "method std::thread::JoinHandle::join") 方法，可以用来连接新生成的线程。如果生成的线程发生 panics，那么 [`join`](https://rustwiki.org/zh-CN/std/thread/struct.JoinHandle.html#method.join "method std::thread::JoinHandle::join") 将返回一个 [`Err`](https://rustwiki.org/zh-CN/std/result/enum.Result.html#variant.Err "variant std::result::Result::Err")，其中包含了提供给 [`panic!`](https://rustwiki.org/zh-CN/std/macro.panic.html "macro std::panic") 的参数。

如果连接句柄被丢弃了，则新生成的线程将被隐式地 _分离_。 在这种情况下，新生成的线程可能不再被连接。 (程序有责任最终连接它创建的线程，或者将它们分离; 否则，将导致资源泄漏。)

此调用将使用 [`Builder`](https://rustwiki.org/zh-CN/std/thread/struct.Builder.html "struct std::thread::Builder") 的默认参数创建一个线程，如果要指定栈大小或线程名称，请使用此 API。

正如您在 `spawn` 的签名中看到的那样，对于赋予 `spawn` 的闭包及其返回值都有两个约束，让我们对其进行解释：
### 约束分析
- `'static` 约束意味着**闭包及其返回值必须具有整个程序执行的生命周期**。这是因为**线程可以比它们被创建时的生命周期更长**。
    
    确实，如果线程以及它的返回值可以比它们的调用者活得更久，我们需要确保它们以后仍然有效，并且因为我们不能知道它什么时候返回，因此需要使它们直到程序结束时尽可能有效，因此是 `'static` 生命周期。
    - 使用&5 去move，主线程可以在子线程之前退出，5就是一个悬垂指针

    
- [`Send`](https://rustwiki.org/zh-CN/std/marker/trait.Send.html "trait std::marker::Send") 约束是因为**闭包需要通过值从产生它的线程传递到新线程**。它的返回值将需要从新线程传递到它被 `join` 的线程。 提醒一下，[`Send`](https://rustwiki.org/zh-CN/std/marker/trait.Send.html "trait std::marker::Send") 标记 trait 表示从一个线程传递到另一个线程是安全的。[`Sync`](https://rustwiki.org/zh-CN/std/marker/trait.Sync.html "trait std::marker::Sync") 表示将引用从一个线程传递到另一个线程是安全的。

	- 如果使用rc，就会报错，因为它没有实现send，当你增加计数时，应该保证，这是增加该计数的唯一线程，减少计数不是为并行设计的
	- 返回值也需要实现，返回值需要从一个线程转移到另一个线程
