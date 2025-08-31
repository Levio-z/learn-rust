Rust 中，**唯一允许安全 Rust 中内部可变性（interior mutability）**的类型。
- 它告诉编译器：“我知道我在安全地修改这块内存，请允许我在 `&self` 中做可变操作。”
- 是所有 `Cell`、`RefCell`、`Mutex` 等可变封装的底层核心。
### 编译器优化
#### 只读引用导致的优化假设示例
```rust
fn example(data: &i32) -> i32 {
    let a = *data;
    let b = *data;
    a + b
}

```
编译器优化（基于只读假设）

编译器会假设 `*data` 在两次访问中不会变，所以可以只读一次缓存结果：
```
load data into register r1
a = r1
b = r1
return a + b

```
这就是**基于只读假设的缓存和指令重排序**。
#### 违反假设的“隐藏”写操作（如果存在）
如果底层数据在两次读取之间被**外部修改**了（例如多线程，或者通过某种“隐藏的写”修改了 `data`），但没有告诉编译器，结果就不对了
```rust
// 伪代码：不安全地修改不可变引用指向的数据
unsafe {
    let p = data as *const i32 as *mut i32;
    *p = 42;
}

```
编译器依然基于只读假设优化，没检测到修改，最终结果会出错，造成**未定义行为（UB）**。
#### `UnsafeCell` 如何改变这一切
`UnsafeCell` 是“**告诉编译器这个数据可能被修改的唯一合法方式**”。

示例：
```rust

use std::cell::UnsafeCell;

struct Wrapper {
    data: UnsafeCell<i32>,
}

impl Wrapper {
    fn new(val: i32) -> Self {
        Wrapper { data: UnsafeCell::new(val) }
    }

    fn get(&self) -> i32 {
        unsafe { *self.data.get() }
    }

    fn set(&self, val: i32) {
        unsafe { *self.data.get() = val; }
    }
}

fn example(wrapper: &Wrapper) -> i32 {
    let a = wrapper.get();
    // 这里可以安全地修改
    wrapper.set(10);
    let b = wrapper.get();
    a + b
}

```
#### 编译器处理
- 因为 `data` 在 `UnsafeCell` 中，编译器**不能假设 `data` 只读**。
- **每次访问都必须重新读取内存，不能缓存之前的值。**
- 编译器会在读写操作间插入内存屏障或防止重排序，保证数据修改能被正确看到。

#### 如果自己使用裸指针修改值
- **你必须确保**：该内存当前**没有任何不可变引用（`&T`）或其他线程同时读取**。
- 换言之，在修改内存时，不能有任何别的地方基于该内存做只读访问，否则缓存和重排导致读取到脏数据。
- 同时不能有任何可变借用（`&mut T`）冲突。