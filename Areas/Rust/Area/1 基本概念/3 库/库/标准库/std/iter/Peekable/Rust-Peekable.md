用于增强迭代器功能的适配器类型。它通过提供 `peek()` 和 `peek_mut()` 方法，使得在迭代过程中可以在不消耗元素的情况下查看下一个元素，从而在需要预览下一个元素的场景中非常有用。
### 一、Peekable 的定义与创建

`Peekable` 是通过调用迭代器的 `peekable()` 方法创建的：
`let iter = some_iterator.peekable();`
这将返回一个 `Peekable<I>` 类型的迭代器，其中 `I` 是原始迭代器的类型。

---

### 二、主要方法

#### 1. `peek()`

返回一个对下一个元素的不可变引用，但不会消耗该元素。

```rust
fn main() {
    let mut iter = vec![1, 2, 3].into_iter().peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.peek(), Some(&2));
}
```

需要注意的是，`peek()` 返回的是对元素的引用，因此可能会出现双重引用的情况，例如 `&&1`。

#### 2. `peek_mut()`

返回一个对下一个元素的可变引用，同样不会消耗该元素。

```rust
fn main() {
let mut iter = vec![1, 2, 3].into_iter().peekable(); 
if let Some(peek) = iter.peek_mut() {  
*peek *= 2;
} 
assert_eq!(iter.next(), Some(2));
}
```

这对于在迭代过程中修改元素值非常有用。

---

### 三、使用场景

- **解析器实现**：在编写解析器时，常常需要查看下一个元素以决定当前元素的处理方式。`Peekable` 可以在不消耗元素的情况下实现这一点。
    
- **查找模式匹配**：在处理数据流时，可能需要查找特定的模式，例如查找连续的换行符 `\r\n`。使用 `peek()` 可以在不消耗元素的情况下查看下一个元素，以便进行模式匹配。
    
- **提前检查**：在处理数据流时，可能需要在处理当前元素之前检查下一个元素的值，以决定是否需要进行某些操作。`peek()` 提供了这种能力。