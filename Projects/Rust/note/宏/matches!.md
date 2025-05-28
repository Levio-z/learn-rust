`matches!` 宏会自动解引用（Deref Coercion），所以即使 `c` 是 `&char`，它仍然可以正确匹配 `char` 字面量（如 `'a'`、`'0'` 等）。这是因为：
### **1. Deref Coercion（自动解引用）**

Rust 会自动对引用类型进行解引用，使其能匹配目标类型。例如：
```rust
let c: &char = &'a';
matches!(c, 'a');  // 等价于 matches!(*c, 'a')
```
### **2. `matches!` 宏的实现**

`matches!` 的定义类似于：

```rust
macro_rules! matches {
    ($expression:expr, $pattern:pat) => {
        match $expression {
            $pattern => true,
            _ => false,
        }
    };
}
```

由于 `match` 本身支持自动解引用，所以 `matches!` 也能正确处理 `&char`。
### 3. 示例验证
```rust
fn main() {
    let c: &char = &'a';

    // 以下方式等效，都能正确匹配
    println!("{}", matches!(c, 'a'));               // true
    println!("{}", matches!(*c, 'a'));              // true（显式解引用）
    println!("{}", matches!(c, 'a'..='z'));         // true
    println!("{}", matches!(c, 'A'..='Z' | '0'..='9' | '_'));  // false
}
```