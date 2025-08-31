[`core::sync::atomic`]
`AtomicBool` 是 Rust 提供的一种**原子类型**，位于核心库中的 [`core::sync::atomic`] 模块，表示一个可以被多个线程**无锁（lock-free）地并发访问与修改的布尔值**。它的作用类似于 `bool`，但具备线程安全特性，适用于构建低层并发原语。
### 核心原理
`AtomicBool` 提供**原子操作**语义，避免了数据竞争。比如在多线程环境中：
- 普通的 `bool` 在被多个线程同时读取和写入时需要加锁（如 `Mutex<bool>`）；
- 而 `AtomicBool` 保证所有操作都是**原子的（不可中断）**，不需要锁；
这对于实现诸如**一次性初始化（如上文的 `LazyInit`）**、**信号标志**、**任务状态标记**等非常有用。

### 常用方法说明

| 方法                                                  | 说明                                  |
| --------------------------------------------------- | ----------------------------------- |
| `new(val: bool)`                                    | 创建一个新的原子布尔值                         |
| `load(ordering)`                                    | 加载当前布尔值，带内存序                        |
| `store(val, ordering)`                              | 原子地存储一个布尔值                          |
| `swap(val, ordering)`                               | 原子地替换布尔值并返回旧值                       |
| `compare_exchange(expected, new, success, failure)` | CAS 操作：若当前值等于 `expected`，则替换为 `new` |
| `fetch_xor/and/or(...)`                             | 按位运算支持（针对布尔值）                       |
### 🔐 Ordering 内存序语义
- `Ordering::Relaxed`：不保证同步，仅保证原子性（性能最高）；
- `Ordering::Acquire`：**所有在程序顺序中位于该加载操作之后的内存操作（读或写）**，都不会被编译器或 CPU 重排到加载操作之前执行。；
- `Ordering::Release`：保证“前序写”不被重排到后面；
- `Ordering::AcqRel`：结合 Acquire + Release；
- `Ordering::SeqCst`：最强同步语义，强顺序一致性。
在 `LazyInit` 中使用：
```
.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)

```

代表：
- 如果当前是 `false`，就设置为 `true`；
- 成功时使用 `Acquire`（用于同步后续读取）；
- 失败时用 `Relaxed`（不需要额外同步开销）；

### 使用场景总结

|场景|使用方式|
|---|---|
|`Once` / `Lazy` 实现|标记是否初始化|
|`Spinlock` 等原子锁|判断是否上锁|
|`任务取消`、`终止信号`|多线程共享布尔标志位|
|`状态机标志位`|状态是否有效/就绪等|

#### Ordering::Release
- **写屏障（Store-Store Barrier）** 是一种保证：在该屏障之前的所有写操作**必须在该屏障之后的写操作之前被执行（对本核可见）**，即禁止写操作重排序。

- **写屏障本身不直接“刷出”数据**，而是保证指令执行顺序，防止 CPU 乱序执行写指令
```rust
// Thread A (writer)
data = 123;
ready.store(true, Ordering::Release);

// Thread B (reader)
if ready.load(Ordering::Acquire) {
    assert_eq!(data, 123); // ✅ 保证看到初始化过的 data
}

```
-  保证了什么？
	- `data = 123` 写在 `ready.store(...)` 之前；
	- `Release` 保证：一旦其他线程看到 `ready == true`，那么也一定能看到之前的 `data = 123`。
`Release` 语义会被编译成：
- **编译屏障（禁止编译器重排）**
- **CPU 指令屏障（如 x86 的 `sfence`，ARM 的 `dmb ish`）**
- 这些 CPU 指令屏障：
    - **保证写操作的顺序**
    - **使写缓冲区中的写操作“冲刷”到缓存系统，并发出缓存一致性协议的通知**   
		- **写入真正被其他 CPU 看到的时间取决于缓存一致性协议，不是立刻写回主存**。
#### Ordering::Acquire

`Acquire` 是一种**加载（load）操作**的内存序保证，主要用于多线程同步，保证：
- **当前线程对内存的“后续读写操作”不会被执行在该 `load` 之前**，即不会被重排序到 `load` 指令之前。
- 当你用 `Acquire` 加载一个原子变量时，如果该加载成功（比如读取到了另一个线程的 `Release` 存储值），  
	那么当前线程**从此 `load` 指令开始，看到的所有内存操作都是同步的（可见的最新的）**。
```rust
// Thread A (发布者)
data = 42;                          // 普通写
flag.store(true, Ordering::Release); // 发布信号

// Thread B (等待者)
if flag.load(Ordering::Acquire) {   // 获取发布信号
    assert_eq!(data, 42);            // 必须看到最新的 data
}

```
- `flag.store(..., Release)` 保证了之前对 `data` 的写入被“发布”出去。
- `flag.load(..., Acquire)` 保证后续访问 `data` 不会被提前，也就是说不会在检查 `flag` 之前执行。

#### CPU和编译器层面的保证
- **编译器层面**  
    `Acquire` 会阻止编译器把后续读写指令往前提早执行。
- **CPU层面**  
    会插入合适的内存屏障（如 ARM 的 `dmb ish ld`），保证加载操作之前的指令执行完毕，加载操作成功后再执行后续指令。
### Release和Acquire类比图解
可以把 `Release` 视为线程内的“写入打包点”：
```rust
Thread A timeline
[ data = 123 ] --> [ data2 = 456 ] --> [ ready.store(true, Release) ]
                                |
                                v
                           写入打包并发布

```
然后 `Acquire` 相当于线程 B 的“接收点”：
```rust
Thread B timeline
[ if ready.load(Acquire) == true ] --> 安全读取 data 与 data2

```