关键点：`Option::map(self)` **会消费（move）调用者**

`Option<T>` 的 `map` 方法签名是：
```rust
impl<T> Option<T> {
    pub fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U,
    { ... }
}
```
注意： 
- `map` 接收的是 **`self`，也就是调用者的所有权**，而不是引用。
- 调用 `self.next.map(...)`，意味着**试图把 `self.next`（整个 Option）移动到 `map` 中去**。

但 `Option<&mut Node<T>>` 并不实现 `Copy`，它是不可复制的。