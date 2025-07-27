需要 GC 行为（复杂图结构 + 多变 + 不关心 drop）

这是 Rust **不直接支持** 的场景。

不过你可以选择：

#### ✅ `gc` crate：**显式垃圾收集器**

```toml
gc = "0.4"
```

```rust
use gc::{Gc, Finalize, Trace};

#[derive(Trace, Finalize)]
struct Node {
    value: i32,
    next: Option<Gc<Node>>,
}
```

由 `gc` crate 控制生命周期。适合：

-   编写解释器、DSL、图遍历、图编辑器
    
-   没有 Rust 所有权建模能力的结构
    
