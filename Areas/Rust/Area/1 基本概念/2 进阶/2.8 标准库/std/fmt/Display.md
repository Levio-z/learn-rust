`std::fmt::Display`
```rust
write!(f, "IO error: {}", e)
```
- `write!` 是一个宏，用于向实现了 `std::fmt::Write` trait 的目标（这里是 `f`，一个格式化器）写入格式化文本。
- `"IO error: {}"` 是格式字符串，其中 `{}` 会被替换为后面参数 `e` 的格式化输出。
- `e` 是一个错误类型，这里是 `io::Error`，它本身实现了 `Display`，所以可以被格式化输出。
- 总结：这句代码的作用是向格式化输出器 `f` 写入字符串 `"IO error: "` 加上具体的 `io::Error` 的描述文本，实现了自定义错误类型中，`Display` trait 的格式化输出逻辑。