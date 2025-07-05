https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
- 'copy semantics'
	- 默认情况下，变量绑定具有 'move semantics'
	- 如果类型实现了 `Copy`，则它具有 'copy semantics'
	- 唯一的区别是是否允许您在分配后访问 `x`,后台都会复制
- 如何实施 `Copy`
	- 使用 `derive`
	- 手动实施 `Copy` 和 `Clone`
	- 区别
		- derive 策略还将放置一个 Copy 绑定类型参数
```rust
#[derive(Clone)]
struct MyStruct<T>(T);

impl<T: Copy> Copy for MyStruct<T> { }
```
这并不总是希望的