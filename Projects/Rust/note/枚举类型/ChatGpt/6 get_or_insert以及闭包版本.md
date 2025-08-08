`get_or_insert(value: T) -> &mut T`
- **作用**：如果为 `None`，插入 value 并返回引用；否则返回已有值的引用
- **示例**：
```rust
let mut x = None;
x.get_or_insert(10); // x = Some(10)
```

`get_or_insert_with(|| -> T)`

**懒执行版本**，只有在 `None` 时才调用闭包，非常适合构造代价昂贵的默认值。