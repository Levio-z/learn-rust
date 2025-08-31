**`None` 和 `Some` 分支的顺序不会影响生命周期的判断**。在 Rust 中，`match` 的所有分支是**互斥且独立的作用域**，编译器会对每个分支**单独分析**其是否使用了借用，因此顺序不影响借用是否结束。
```rust
fn add(&mut self, handler: Box<dyn Handler>) {
    match &mut self.next {
        None => {
            // ✅ 此处没有使用之前的可变借用
            self.next = Some(Box::new(HandlerNode::new(handler)));
        }
        Some(next) => {
            next.add(handler); // ✅ 可变借用仅在此分支内生效
        }
    }
}

```