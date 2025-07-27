### 源码
```rust
    pub fn as_deref(&self) -> Option<&T::Target>

    where

        T: Deref,

    {

        self.as_ref().map(|t| t.deref())
    }
```

`as_deref` 和 `as_deref_mut` 是 Rust 标准库中 `Option` 类型提供的两个便捷方法，从 **Rust 1.40** 开始稳定。
[2 as_ref](2%20as_ref.md)
它们的设计初衷是：  

让你**更方便地将 `Option<T>` 转换为 `Option<&U>` 或 `Option<&mut U>`，其中 `T: Deref<Target = U>`**，即适用于如 `Box<T>`、`Rc<T>`、`Arc<T>`、`String`、`Vec<T>` 这样的智能指针或拥有类似行为的类型。
### 使用场景概览
举个简单的例子：
```rust
let opt_box: Option<Box<String>> = Some(Box::new("hello".to_string()));
let opt_ref: Option<&str> = opt_box.as_deref();
```
`as_deref` 会将 `Option<Box<String>>` 转换成 `Option<&str>`，因为：
- `Box<String>` 实现了 `Deref<Target=String>`
- `String` 实现了 `Deref<Target=str>`
### 函数签名与定义
```rust
impl<T> Option<T> {
    pub fn as_deref(&self) -> Option<&T::Target>
    where
        T: Deref,
    ```

```rust
    pub fn as_deref_mut(&mut self) -> Option<&mut T::Target>
    where
        T: DerefMut,
}

```

简析：

- `as_deref`：接收 `&Option<T>`，返回 `Option<&T::Target>`
- `as_deref_mut`：接收 `&mut Option<T>`，返回 `Option<&mut T::Target>`

关键点是：自动调用 `Deref` 或 `DerefMut` 来获取内部的 `&Target` 或 `&mut Target`