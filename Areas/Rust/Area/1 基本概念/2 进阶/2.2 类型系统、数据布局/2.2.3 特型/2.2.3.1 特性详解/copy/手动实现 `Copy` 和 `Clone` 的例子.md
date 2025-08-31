1. 基础类型包装
2. 引用类型包装（`&T`）
3. 使用 `PhantomData` 的零大小类型
4. 部分成员为 `Copy`，部分不是
5. 带生命周期的类型
* * *

### ✅ 示例 1：基础类型包装（派生 `Copy`）

```rust
#[derive(Copy, Clone, Debug)]
struct Wrapper(u32);
```

这是最基本的形式。因为 `u32` 是 `Copy` 的，所以 `Wrapper` 也可以 `Copy`。

* * *

### ✅ 示例 2：手动实现 `Copy` + 引用类型（无需 `T: Copy`）

```rust
#[derive(Debug)]
struct RefHolder<'a, T>(&'a T);

// &T 是 Copy，即使 T 不是 Copy，也没关系
impl<'a, T> Copy for RefHolder<'a, T> {}
impl<'a, T> Clone for RefHolder<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn main() {
    let x = String::from("hello");
    let holder = RefHolder(&x);
    let copy = holder; // 拷贝的是引用，不是 x 的值
    println!("{:?}", copy);
}
```

* `&T` 是 Copy；
* 即使 `T = String`（不是 Copy），也不会报错；
* derive 就做不到这一点，因为它会强制要求 `T: Copy`。
```rust
#[derive(Debug)]
struct RefPair<'a>(&'a u32, &'a u32);

// 两个引用都是 Copy，可实现 Copy
impl<'a> Copy for RefPair<'a> {}
impl<'a> Clone for RefPair<'a> {
    fn clone(&self) -> Self {
        *self
    }
}

```
- 同上
* * *

### ✅ 示例 3：`PhantomData` 实现 Copy

```rust
use std::marker::PhantomData;

#[derive(Debug)]
struct TypeMarker<T> {
    _marker: PhantomData<T>,
}

// PhantomData 不占空间，本质是零大小类型，安全 Copy
impl<T> Copy for TypeMarker<T> {}
impl<T> Clone for TypeMarker<T> {
    fn clone(&self) -> Self {
        *self
    }
}
```

* 通常用于“占位类型参数”或“标记所有权”；
* 不实际拥有 `T`，所以不需要 `T: Copy`；
* `#[derive(Copy, Clone)]` 会失败，手动实现则成功。
    

* * *

### ✅ 示例 4：结构体部分成员为 `Copy`，部分不是 —— ❌ 不允许 Copy

```rust
struct Mixed {
    a: u32,         // Copy
    b: String,      // 非 Copy
}

// 下面这两行会报错，因为 String 不是 Copy
// impl Copy for Mixed {}
// impl Clone for Mixed { fn clone(&self) -> Self { *self } }
```

* 如果你想让 `Mixed` 可复制，就必须用 `Clone` 并手动拷贝 `b.clone()`，不能用 `Copy`。
    

* * *
## ✅ 小结：什么时候推荐手动实现？

| 场景 | 是否适合手动实现 Copy |
| --- | --- |
| 类型成员全是 `Copy` 类型 | ✅（但推荐 `derive`） |
| 包含引用类型成员（如 `&T`） | ✅（手动实现更灵活） |
| 包含 `PhantomData` 类型成员 | ✅（通常 derive 会误判） |
| 包含非 Copy 成员（如 `String`） | ❌（无法 Copy，只能 Clone） |

* * *