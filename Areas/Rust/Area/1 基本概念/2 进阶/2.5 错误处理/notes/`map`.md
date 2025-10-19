### Rust `map` 方法详解

---

#### 方法签名

```rust
impl<T> Option<T> {
    pub fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U;
}
```

**内部逻辑**

```rust 
match self { 
	Some(val) => Some(f(val)),
	None => None, 
}
```
**关键点**

1. **闭包返回类型**：`U` (一个普通的值)。 

2. **装箱**：`map` 会自动将闭包返回的 `U` 重新包装成 `Some(U)`。

---
#### 基本用法

```rust
let opt = Some(5);
let result = opt.map(|x| x * 2); // Some(10)

let none_opt: Option<i32> = None;
let result2 = none_opt.map(|x| x * 2); // None
```

---

#### 与 `and_then` 区别

- `map`：闭包返回普通值，返回 `Option<U>` 自动包装。`f: T -> U`
    
- `and_then`：闭包返回 `Option<U>`，用于链式组合。`f: T -> Option<U>`
    

示例对比：

```rust
let opt = Some(2);
let map_result = opt.map(|x| x * 2); // Some(4)
let and_then_result = opt.and_then(|x| if x > 0 { Some(x * 2) } else { None }); // Some(4)
```

---

#### 链式使用

```rust
let opt = Some(3);
let result = opt
    .map(|x| x + 1)
    .map(|y| y * 2); // Some(8)
```

- 多次 `map` 可以连续对值进行操作，`None` 会自动短路。
    

---

#### 小结

- `map` 用于对 Option 内的值做映射操作。
    
- 如果 Option 是 `None`，map 不会执行闭包，直接返回 `None`。
    
- 常用于安全链式处理、数据转换和组合。