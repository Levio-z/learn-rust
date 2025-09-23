
### Rust std
- **Thread（线程）**：执行的代理（即一个独立的执行流）。
	- **安全性**：
		- 要求闭包是 `'static` 生命周期（而不是函数指针），确保线程中不会捕获悬空引用。线程可能比创建它的作用域活得更久。如果闭包捕获了局部变量（比如栈上的引用），线程一旦延迟执行，就可能访问悬空指针。`'static` 限制保证闭包捕获的数据要么是 **拥有所有权的值**（被 move 进线程），要么是 **真正 `'static` 的引用**（例如全局常量）。这就防止了悬垂引用问题。
		- `join` 返回的是带类型的句柄（typed join handle），避免未定义行为。我们可以用类型安全的方式拿到线程返回值。
			- 原来：**类型不安全**：你必须自己在调用方和线程内部约定好返回值的实际类型，然后在 `void*` 和真实类型之间做强制转换。**容易出错**：如果你转换成了错误的类型，就会产生 **未定义行为**（UB）。
			- 仙子啊：**调用方无需记住类型或做转换**，编译器会自动推导 `T`。**避免未定义行为**，因为你不可能把 `JoinHandle<i32>` 当成 `JoinHandle<String>` 使用。
- **Scoped thread（作用域线程）**：限制线程的生命周期在某个作用域内。
	- **动机**：允许安全地共享非 `'static` 数据（如局部变量的引用）。
	- **安全性**：必须保证线程在作用域结束前被 `join`，避免悬垂引用。
	- **好处**：而是让线程直接**借用外部变量**。
- **Arc（原子引用计数）**：在多个线程间不可变共享数据的方式。
	- **安全性**：只实现 `Deref`，而不是 `DerefMut`，因此只能不可变借用，防止数据竞争。
- **Send**：类型可以被安全地转移到其他线程。
	- **实现者**：`usize`, `&usize`, `Arc<T>`, `&Arc<T>`
	- **不实现者**：`Rc<T>`, `&Rc<T>`（因为 `Rc` 非线程安全）。
- **Sync**：类型可以在多个线程中同时被安全访问。
	- **实现者**：`usize`, `Arc<T>`
	- **不实现者**：`Rc<T>`
- **性质**：
    - 当且仅当 `&T: Send` 时，`T: Sync`。
    - 也就是说：如果 `T` 的共享引用 `&T` 能被安全地转移到另一个线程，那么 `T` 本身就是 `Sync`。

[Spawn方法](../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.8%20标准库/std/thread/Spawn/Spawn方法.md)
- Scoped thread example: [https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=24a32b6b8e806e6139cca20e001b6a70](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=24a32b6b8e806e6139cca20e001b6a70)


- Into_par_iter example: [https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=5112d3cc4cb42c82f4a2b7bb5d98951e](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=5112d3cc4cb42c82f4a2b7bb5d98951e)
### parking_lot
`parking_lot` 是 Rust 生态中一个广泛使用的高性能同步原语（synchronization primitives）库，它为 **锁（Mutex, RwLock）、条件变量（Condvar）、信号量（Semaphore）、Barrier** 等并发工具提供了比标准库更高效、更灵活的实现。其核心理念是通过 **无锁队列 + 操作系统阻塞原语** 的组合，避免了标准库 `std::sync` 在很多情况下的性能瓶颈。

Rust 标准库 `std::sync::Mutex` 等同步原语依赖于 **OS 原语（如 pthread_mutex）**，这带来几个问题：

1. **性能开销大**：直接调用内核的锁和唤醒系统调用。
2. **可扩展性差**：标准库 API 受限，功能相对单一。
3. **公平性问题**：标准库锁实现对唤醒顺序没有更多优化。

#### parking_lot 的核心原理
##### 1. Parking / Unparking 模型
- **Park（停车）**：当线程无法获取锁时，将其标记为“停车”，并进入等待队列。
- **Unpark（解锁唤醒）**：当锁释放时，唤醒等待队列中的线程（通常是 FIFO 或公平策略）。

这种机制类似于 **自旋锁 + futex** 的混合体：
- 首先在用户态自旋一段时间（避免系统调用）。
- 若仍失败，则调用 OS 提供的底层阻塞机制（如 Linux futex、Windows SRWLock/Condvar）。
##### 2. 等待队列（Wait Queue）
- 每个锁对象关联一个等待队列。
- 队列用无锁数据结构实现，避免额外锁竞争。
- 线程进入队列时挂起，等待 `unpark` 唤醒。
##### 3. `parking_lot_core`
- `parking_lot` 本身只是 API 层。
- 其核心逻辑在 `parking_lot_core` crate：
    - 维护线程哈希表（基于地址映射）。
    - 提供 `park` / `unpark_one` / `unpark_all` 等底层接口。
`parking_lot` 的目标是：**用用户态的等待队列管理大多数锁竞争，将真正的阻塞/唤醒交给极少数情况的系统调用，从而显著提升性能**。
#### `Lock<L: RawLock, T>`
- **定义与作用**：  
    表示一个拥有 `T` 的对象，该对象受 `L` 类型的锁保护。
- **保证**：  
    `T` 对象不会被并发访问（注意：这里指对象访问，而不是代码块访问）。
- **示例**：
    - `Lock<SpinLock, Vec<usize>>` —— 用自旋锁保护一个 `Vec`
    - `Lock<ClhLock, &'t TLS>` —— 用 CLH 锁保护一个线程局部存储的引用
- **属性**：
     如果 `T` 是 `Send`，则 `Lock<L, T>` 同时是 `Send` 和 `Sync`（仅当 `T` 可跨线程传递时才有意义）。
        

---

#### `LockGuard<'s, L: RawLock, T>`

- **定义与作用**：  
    用于表示锁已经被成功获取。
    
- **保证**：
    
    - 锁被持有
        
    - 可以通过 `Deref` / `DerefMut` 安全访问 `T`
        
- **RAII 特性**：  
    当 `LockGuard` 被丢弃（`drop`）时自动释放锁
    
- **属性**：
    
    - 如果 `T` 是 `Send`，则 `LockGuard` 是 `Send`
        
    - 如果 `T` 是 `Sync`，则 `LockGuard` 是 `Sync`
        
    - 即它是对底层数据的透明访问器。


#### **标准库（std）**
- **`Mutex`**：互斥锁，支持多种实现策略，保证同一时间只有一个线程访问数据。
>允许你以唯一可变或不可变的方式，可变的访问底层数据
- **`Condvar`**：条件变量，用于等待某个事件（条件）发生。
    - **安全性**：`Condvar::wait()` 接收 `&mut MutexGuard`，禁止在等待期间重复使用被保护的数据。
- **`RwLock`**：读写锁，允许**多个读者**同时访问，或**单个写者**独占访问。
    
https://docs.rs/parking_lot/0.12.4/parking_lot/type.Mutex.html
---

#### **crossbeam**

- **`Channel`**：线程间发送/接收值的通道。
    
- **`CachePadded`**：对齐到 128 字节边界。
    
    - **动机**：避免“伪共享”（false sharing）问题，提高多线程性能。
        

---

#### **rayon**

- **`into_par_iter`**：并行执行一个函数对每个元素进行操作。
    
- **动机**：让并行处理变得简单易用。