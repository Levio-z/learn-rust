`IntoIterator` trait 定义如下（简化版）：
```rust
pub trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter;
}

```
- 关联类型 `IntoIter`
	**必须**实现 `Iterator` trait
	且其 `Item` 类型与 `IntoIterator::Item` 相同。
- 因此，**实现 `IntoIterator` 的时候，`IntoIter` 类型必须实现 `Iterator`，且该迭代器的 `Item` 类型必须匹配声明的 `Item` 类型。**
- 继承了 `IntoIterator` trait 的定义约束，因此在这里不必再重复写出 `IntoIter: Iterator<Item=Self::Item>`。