### new
- **不是的**，`f` 不是必须是 `const fn`，而只要满足 `FnOnce() -> T` 的 trait bound 就可以。
```rust
/// Creates a new lazy value with the given initializing function.

    ///

    /// # Examples

    ///

    /// ```

    /// use std::sync::LazyLock;

    ///

    /// let hello = "Hello, World!".to_string();

    ///

    /// let lazy = LazyLock::new(|| hello.to_uppercase());

    ///

    /// assert_eq!(&*lazy, "HELLO, WORLD!");

    /// ```
#[inline]

    #[stable(feature = "lazy_cell", since = "1.80.0")]

    #[rustc_const_stable(feature = "lazy_cell", since = "1.80.0")]

    pub const fn new(f: F) -> LazyLock<T, F> {

```
### 方法
#### new
```rust
    /// Creates a new lazy value with the given initializing function.

    ///

    /// # Examples

    ///

    /// ```

    /// use std::sync::LazyLock;

    ///

    /// let hello = "Hello, World!".to_string();

    ///

    /// let lazy = LazyLock::new(|| hello.to_uppercase());

    ///

    /// assert_eq!(&*lazy, "HELLO, WORLD!");

    /// ```

    #[inline]

    #[stable(feature = "lazy_cell", since = "1.80.0")]

    #[rustc_const_stable(feature = "lazy_cell", since = "1.80.0")]

    pub const fn new(f: F) -> LazyLock<T, F> {

        LazyLock { once: Once::new(), data: UnsafeCell::new(Data { f: ManuallyDrop::new(f) }) }

    }
```
- `f` 不是必须是 `const fn`，而只要满足 `FnOnce() -> T` 的 trait bound 就可以。

#### into_inner
```rust
    pub fn into_inner(mut this: Self) -> Result<T, F> {

        let state = this.once.state();

        match state {

            ExclusiveState::Poisoned => panic_poisoned(),

            state => {

                let this = ManuallyDrop::new(this);

                let data = unsafe { ptr::read(&this.data) }.into_inner();

                match state {

                    ExclusiveState::Incomplete => Err(ManuallyDrop::into_inner(unsafe { data.f })),

                    ExclusiveState::Complete => Ok(ManuallyDrop::into_inner(unsafe { data.value })),

                    ExclusiveState::Poisoned => unreachable!(),

                }

            }

        }

    }

```

- 消耗（consume）整个 `LazyLock`，返回内部存储的值 `T` 或初始化闭包 `F`（如果未初始化）
##### 场景
- 提前判断：你可能不想使用 `*lazy` 去触发初始化，而是先判断它有没有初始化。
- 手动析构管理：获取 `T` 后，你可以自行决定何时 drop。
- 性能敏感路径中避免自动初始化。
- 测试场景：观察 Lazy 是否真的初始化过。

#### Deref
```rust
#[stable(feature = "lazy_cell", since = "1.80.0")]

impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {

    type Target = T;

  

    /// Dereferences the value.

    ///

    /// This method will block the calling thread if another initialization

    /// routine is currently running.

    ///

    #[inline]

    fn deref(&self) -> &T {

        LazyLock::force(self)

    }

}

```
#### get_mut
```rust
  #[inline]

    #[unstable(feature = "lazy_get", issue = "129333")]

    pub fn get_mut(this: &mut LazyLock<T, F>) -> Option<&mut T> {

        // `state()` does not perform an atomic load, so prefer it over `is_complete()`.

        let state = this.once.state();

        match state {

            // SAFETY:

            // The closure has been run successfully, so `value` has been initialized.

            ExclusiveState::Complete => Some(unsafe { &mut this.data.get_mut().value }),

            _ => None,

        }
```
#### get
```rust
/// Returns a reference to the value if initialized, or `None` if not.

    ///

    /// # Examples

    ///

    /// ```

    /// #![feature(lazy_get)]

    ///

    /// use std::sync::LazyLock;

    ///

    /// let lazy = LazyLock::new(|| 92);

    ///

    /// assert_eq!(LazyLock::get(&lazy), None);

    /// let _ = LazyLock::force(&lazy);

    /// assert_eq!(LazyLock::get(&lazy), Some(&92));

    /// ```

    #[inline]

    #[unstable(feature = "lazy_get", issue = "129333")]

    pub fn get(this: &LazyLock<T, F>) -> Option<&T> {

        if this.once.is_completed() {

            // SAFETY:

            // The closure has been run successfully, so `value` has been initialized

            // and will not be modified again.

            Some(unsafe { &(*this.data.get()).value })

        } else {

            None

        }

    }

}
```
#### force
```rust
    #[inline]

    #[stable(feature = "lazy_cell", since = "1.80.0")]

    pub fn force(this: &LazyLock<T, F>) -> &T {

        this.once.call_once(|| {

            // SAFETY: `call_once` only runs this closure once, ever.

            let data = unsafe { &mut *this.data.get() };

            let f = unsafe { ManuallyDrop::take(&mut data.f) };

            let value = f();

            data.value = ManuallyDrop::new(value);

        });

  

        // SAFETY:

        // There are four possible scenarios:

        // * the closure was called and initialized `value`.

        // * the closure was called and panicked, so this point is never reached.

        // * the closure was not called, but a previous call initialized `value`.

        // * the closure was not called because the Once is poisoned, so this point

        //   is never reached.

        // So `value` has definitely been initialized and will not be modified again.

        unsafe { &*(*this.data.get()).value }

    }
```
- `&LazyLock<T, F>`：接收一个对 `LazyLock` 的引用，不需要可变。
- `-> &T`：返回初始化后的内部值 `T` 的不可变引用。
这是一个 **静态方法**（不是 `self` 方法），通过传入引用来使用。
```rust
this.once.call_once(|| {
    let data = unsafe { &mut *this.data.get() };
    let f = unsafe { ManuallyDrop::take(&mut data.f) };
    let value = f();
    data.value = ManuallyDrop::new(value);
});

```
 步骤 1：只执行一次的闭包逻辑
- `this.once.call_once(...)` 是同步原语 `Once` 的入口：
步骤 2：解引用 `UnsafeCell` 获取可变引用
- 然后返回unsafesell获取裸指针，转为不可变引用
-  这个动作是 **移动所有权** —— 你得到了内部的 `T`，并拥有它。
- 但是，`ManuallyDrop<T>` 自身的内存还保留着原来 `T` 的字节（未清理、不自动 drop）。
步骤 3：提取闭包（f）并运行
- 从 `ManuallyDrop<F>` 中取出 `f`，这是惰性初始化的函数（比如 `|| 92`）。
- 立即运行这个闭包生成 `T`。
存入 `value` 字段（也是 ManuallyDrop）
- data.value = ManuallyDrop::new(value);
	- 将真正初始化的值写入结构体中。
#### drop
```rust
#[stable(feature = "lazy_cell", since = "1.80.0")]

impl<T, F> Drop for LazyLock<T, F> {

    fn drop(&mut self) {

        match self.once.state() {

            ExclusiveState::Incomplete => unsafe { ManuallyDrop::drop(&mut self.data.get_mut().f) },

            ExclusiveState::Complete => unsafe {

                ManuallyDrop::drop(&mut self.data.get_mut().value)

            },

            ExclusiveState::Poisoned => {}

        }

    }

}
```