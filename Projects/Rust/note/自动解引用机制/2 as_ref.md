### 源码
```rust
    pub const fn as_ref(&self) -> Option<&T> {

        match *self {

            Some(ref x) => Some(x),

            None => None,

        }

    }
```


```rust
Some(ref x) => Some(x)
```
其中的 `ref` 是 **模式匹配中的一种修饰符**，不是关键字 `&` 的别名，而是用于**在匹配中创建引用**。

### 作用
回答你的问题：`ref` 是什么？
在 Rust 中，如果你在模式匹配中写：
```rust
Some(ref x)
```
意思是：

> “**匹配一个 `Some`，但我不想移动其中的值，而是想**获取它的引用”。

```rust
pub const fn as_ref(&self) -> Option<&T> {
    match *self {
        Some(ref x) => Some(x),
        None => None,
    }
}
```
### 拆解：
- `self` 是 `&&Option<T>` 类型
- `*self` 解引用一次变成 `Option<T>`
- `Some(ref x)` 不 move `T`，而是构造 `&T`
- `Some(x)` 返回的是 `Option<&T>`

这是 `Option<T>` 提供的非常经典的 `as_ref()` 方法实现方式：把一个 `Option<T>` 转成一个 `Option<&T>`。