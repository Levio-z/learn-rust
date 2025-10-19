```
pub fn from_utf8_lossy(v: &[u8]) -> Cow<'_, str>{

}
```
### 函数定义与作用
`String::from_utf8_lossy` 是 Rust 标准库中提供的一个函数，用于**将任意字节切片 (`&[u8]`) 转换为 UTF-8 字符串**。
- 如果字节切片中全部是合法的 UTF-8 字节序列 → 返回一个 **借用的字符串切片 (`Cow::Borrowed`)**，避免额外分配。
- 如果存在非法的 UTF-8 序列 → 会把这些非法部分替换为 **Unicode 替换字符 `U+FFFD (�)`**，并返回 **堆上分配的新 `String` (`Cow::Owned`)**。
这就是为什么函数的返回值是 `Cow<'_, str>` —— 能在“零开销（Borrowed）”与“需要修复（Owned）”之间自动切换。

### 拓展
**性能敏感场景**：如果你确定输入一定是合法 UTF-8，可以使用
- `String::from_utf8`（返回 `Result<String, FromUtf8Error>`，避免额外替换开销）