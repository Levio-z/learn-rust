- **作用**：将 `Option` 设为 `Some(value)` 并返回其可变引用
- **优点**：等价于 `*opt = Some(v); opt.as_mut().unwrap()` 的组合式简写
```rust
let mut x = None;
x.insert(42).add_assign(1); // x 现在是 Some(43)
```
