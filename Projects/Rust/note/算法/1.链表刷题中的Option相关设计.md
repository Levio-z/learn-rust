| 方法                    | 作用                          |
| --------------------- | --------------------------- |
| `.unwrap()`           | 取出 Some 内值，None 时 panic     |
| `.unwrap_or(default)` | None 时用 `default` 替代        |
| `.as_ref()`           | 获得 `Option<&T>`（不移动所有权）     |
| `.as_mut()`           | 获得 `Option<&mut T>`（可变引用访问） |
| `.take()`             | 把内部值取出来并置为 None             |
### 为什么用 `Box<ListNode>` 而不是直接 `ListNode`
`ListNode` 这种递归定义：
```rust
struct ListNode {
    val: i32,
    next: Option<Box<ListNode>>,
}

```
如果直接写成：
```rust
struct ListNode {
    val: i32,
    next: Option<ListNode>, // ❌ 编译不过
}

```
会造成无限展开（无限大）。
### 为什么用 `Option<Box<ListNode>>`
如果你想表示：

- 链表的结束点（即没有下一个节点）
- 需要用 `None` 来标志终止，否则链表会无限循环下去。
#### 为什么 Java 不需要 Option/Optional**

在 Java 中：

- `next` 可以是 `null`。
    
- 直接用 `null` 表示链表结束。
    
`if (node.next == null) {     // 链表结束 }`

Rust 中：

- **没有 null**。
    
- 用 `Option<T>` 显式表达可空性。