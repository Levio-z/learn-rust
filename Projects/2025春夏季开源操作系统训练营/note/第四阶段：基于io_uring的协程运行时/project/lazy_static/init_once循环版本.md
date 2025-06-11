```rust
pub fn init_once(&self, data: T) -> &T {
    // 循环直到成功设置 inited = true
    loop {
        match self.inited.compare_exchange_weak(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                // 安全写入数据
                unsafe { (*self.data.get()).as_mut_ptr().write(data) };
                return unsafe { self.force_get() };
            }
            Err(true) => panic!("Already initialized"),
            Err(false) => continue, // 伪失败，重试
        }
    }
}

```
`call_once` 循环版
```rust
pub fn call_once<F>(&self, f: F) -> Option<&T>
where
    F: FnOnce() -> T,
{
    // 尝试获得初始化权限
    let mut f = Some(f); // FnOnce 只能调用一次，因此外包裹一层 Option
    loop {
        match self.inited.compare_exchange_weak(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                unsafe {
                    (*self.data.get()).as_mut_ptr().write(f.take().unwrap()());
                }
                return Some(unsafe { self.force_get() });
            }
            Err(true) => return None,        // 已被初始化
            Err(false) => continue,          // 伪失败，重试
        }
    }
}


```