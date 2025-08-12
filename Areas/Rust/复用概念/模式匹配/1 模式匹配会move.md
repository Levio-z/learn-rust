### 问题核心：match 会尝试 **move**
默认是一个 _by-value_ 的匹配：即尝试将 `self.head` 的值**搬进 `node` 变量里**。但问题是：

- `self.head` 是 `&mut self` 的字段
    
- 而你并没有所有权（你只有 &mut 借用）
- 所以 Rust 会说：

> ❌ 你不能 move 它！它属于别人（整个 self）！