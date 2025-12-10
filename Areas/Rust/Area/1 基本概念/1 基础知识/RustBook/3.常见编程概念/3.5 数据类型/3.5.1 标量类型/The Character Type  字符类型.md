Rust 的 `char` 类型是该语言最原始的字母类型。以下是声明 `char` 值的一些示例：

```rust
fn main() {
    let c = 'z';
    let z: char = 'ℤ'; // with explicit type annotation
    let heart_eyed_cat = '😻';
}
```
请注意，我们指定带有单引号的 `char` 文字，而不是使用双引号的字符串文字。Rust 的 `char` 类型大小为 4 个字节，表示 Unicode 标量值，这意味着它可以表示的不仅仅是 ASCII。重音字母;中文、日文、韩文字符;表情符号;零宽度空格都是 Rust 中有效的 `char` 值。Unicode 标量值的范围从 `U+0000` 到 `U+D7FF` 和 `U+E000` 到 `U+10FFFF`（含）。然而，“字符”在 Unicode 中并不是一个真正的概念，因此您对“字符”是什么的人类直觉可能与 Rust 中的`字符`不匹配。我们将在第 8 章的 [“使用字符串存储 UTF-8 编码文本”](https://rust-book.cs.brown.edu/ch08-02-strings.html#storing-utf-8-encoded-text-with-strings) 中详细讨论此主题。