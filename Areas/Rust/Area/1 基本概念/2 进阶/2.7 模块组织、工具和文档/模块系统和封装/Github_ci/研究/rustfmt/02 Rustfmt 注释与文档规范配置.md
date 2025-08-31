### 1\. `comment_width = 80`
-   **作用**：限制注释的最大宽度为 80 个字符。
-   **效果**：
    -   Rustfmt 会在格式化时，将注释内容控制在 80 个字符以内。    
    -   超过部分会换行，以保证代码在普通编辑器或终端中可读。   
-   **使用场景**：
    -   团队约定注释宽度。
    -   保持代码在窄屏环境下可读性。
---
### 2\. `wrap_comments = true`
-   **作用**：自动换行普通注释（非文档注释）。
-   **效果**：
    -   单行或多行注释超过 `comment_width` 会自动拆成多行。
    -   保持注释整齐，不破坏代码对齐。
-   **示例**：
```rust
// Before
// This is a very long comment that exceeds the maximum width and looks messy in the editor

// After
// This is a very long comment that exceeds the maximum width and looks
// messy in the editor
```
---
### 3\. `format_code_in_doc_comments = true`
-   **作用**：格式化文档注释中的代码块。
-   **效果**：
    -   Rust 的文档注释通常使用三反引号 \`\`\` 包裹代码块。
    -   Rustfmt 会自动格式化这些代码，使其风格与正常代码一致。
-   **示例**：
```rust
/// Example:
/// ```
/// let x =   1+2;
/// ```
///
/// After formatting:
/// ```
/// let x = 1 + 2;
/// ```
```

---
### 4\. `normalize_comments = true`
-   **作用**：统一注释的缩进和空格。
-   **效果**：
    -   删除多余空格。
    -   对齐多行注释。
-   **示例**：

```rust
// Before
//    Some comment with extra spaces

// After
// Some comment with extra spaces
```
---
### 5\. `normalize_doc_attributes = true`
-   **作用**：统一文档注解（doc attributes）排版。
-   **效果**：
    -   Rust 的文档注解通常是 `#[doc = "..."]`。
    -   Rustfmt 会规范化它们的缩进和换行，使文档生成更整齐。
-   **示例**：
```rust
// Before
#[doc="This is a doc string"]
#[doc = "Another line"]

// After
#[doc = "This is a doc string"]
#[doc = "Another line"]
```