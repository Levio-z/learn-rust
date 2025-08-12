`as_mut(&mut self) -> Option<&mut T>`
- **功能**：获取 `Option<T>` 中值的可变引用
- **使用场景**：你想就地修改值，例如对 `Some` 内部的结构体字段赋值
- **示例**：
```rust
let mut x = Some(String::from("hello"));
if let Some(v) = x.as_mut() {
    v.push_str(" world"); // 修改原字符串
}
```
- 避免了值的所有权移动，配合 `match`/`if let` 可以就地安全变更。
- 尤其适合嵌套结构的“局部修改”。
