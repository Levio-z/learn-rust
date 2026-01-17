### 1. 在早期 Rust 中（尤其是 `lazy_static!` 发布时）：
Rust 对 `static` 的初始化有非常严格的限制：
- `static` 只能用 `const fn` 表达式初始化
- 不能有闭包、堆分配、函数调用等**运行时代码**
- 你不能这样写：
```rust
static CONFIG: Lazy<Config> = Lazy::new(|| Config::load()); // ❌ 编译失败（早期）

```
因此：**如果要提供惰性初始化 + 非 const 构造逻辑 + 全局静态变量，就只能使用宏。**
相当于生成了如下 Rust 代码：
```rust
static LOGGER: Lazy<Mutex<Logger>> = Lazy::INIT;

struct Lazy<T> {
    once: Once,
    data: UnsafeCell<MaybeUninit<T>>,
    init_fn: fn() -> T, // 实际就是你写的 `Mutex::new(Logger { .. })`
}

impl<T> Lazy<T> {
    fn get(&self) -> &T {
        self.once.call_once(|| {
            let value = (self.init_fn)();
            unsafe { (*self.data.get()).as_mut_ptr().write(value) };
        });
        unsafe { &*((*self.data.get()).as_ptr() as *const T) }
    }
}

impl<T> Deref for Lazy<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

```
- `static mut` + `*const T` → 用于存放裸指针，避开 borrow checker。
- `Box::into_raw(Box::new(...))` 
	- 将对象 `cfg` 放入堆上，并返回原始裸指针 `*mut T`
	- → 保证值在堆上长期存在（等效 `static` 生命周期）
		- **放弃对资源的 Rust 所有权管理”**，即 Rust 不再帮你 drop 它。
		- 原始指针不会被自动释放，**需要你手动使用 `Box::from_raw(ptr)` 再包装回来以释放内存**。
- `&*CONFIG` → 将裸指针转成 `'static` 引用（带 `unsafe`，读者需小心！）
	- `CONFIG` 是 `static mut`，类型是 `*mut Config`，生命周期为 `'static`。
	- `&*CONFIG` 把裸指针变成了引用。
- `ONCE.call_once` → 保证只有一次写入（惰性初始化）？
### 总结

|方面|特点|
|---|---|
|用户体验|极其优雅，宏用法贴近自然语言|
|底层实现|`unsafe + static mut + Once + Box::into_raw`，需极度小心|
|扩展性|不支持动态初始化参数、重复初始化|
|替代推荐|`once_cell::Lazy` 或 `std::sync::OnceLock`|
|所属时代|在 `const fn` 不成熟时代，它是必要的魔法道具|
### 为什么不移除 lazy_static？

- **兼容性考虑**：已有大量旧项目依赖 `lazy_static!`。
- **宏优势**：支持多个变量同时声明，代码语义清晰：
```rust
lazy_static! {
    static ref A: AType = AType::new();
    static ref B: BType = BType::from(A::clone());
}

```
    
- **不依赖 const fn 的特性演进**：某些场景（如 `no_std` 下）仍可能依赖 unsafe 实现。
### 如何让lazy_static返回的值可以修改
你拿到的是 `&'static Config`，也就是一个共享的不可变引用。Rust 的引用语义严格禁止通过不可变引用去修改值。所以你**不能直接修改 `CONFIG`，除非：**

1. 用 `Mutex<Config>` 包裹，让它具备**可变性 + 线程同步**；
2. 或使用 `UnsafeCell`，并自己保证修改时没有数据竞争（不推荐手写）；
#### 正确的修改姿势
```rust
use lazy_static::lazy_static;
use std::sync::Mutex;

#[derive(Debug)]
struct Config {
    count: usize,
}

lazy_static! {
    static ref CONFIG: Mutex<Config> = Mutex::new(Config { count: 0 });
}

fn update_config() {
    let mut cfg = CONFIG.lock().unwrap();
    cfg.count += 1;
}

```