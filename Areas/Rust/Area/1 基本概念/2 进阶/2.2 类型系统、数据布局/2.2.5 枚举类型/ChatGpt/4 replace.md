- **作用**：用新值替换原值，并返回原来的值
- **对比 `take`**：不是设为 `None`，而是你提供一个新值
```rust
let mut x = Some(1);
let old = x.replace(2); // x = Some(2), old = Some(1)
```