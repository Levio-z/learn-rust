---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
一个字符串模式。
`Pattern` 表示实现该类型的对象可以作为字符串模式，用于在 [`&str`][str] 中进行搜索。
例如，`'a'` 和 `"aa"` 都是模式，它们都能匹配字符串 `"baaaab"` 中索引为 `1` 的位置。
该 Trait 本身充当关联类型 [`Searcher`] 的构造器，而 `Searcher` 则负责在字符串中查找模式出现的具体工作。
根据模式类型的不同，[`str::find`] 和 [`str::contains`] 等方法的行为也会有所变化。下表描述了其中一些行为：

| **模式类型**                 | **匹配条件**                |
| ------------------------ | ----------------------- |
| `&str`                   | 是子字符串                   |
| `char`                   | 字符包含在字符串中               |
| `&[char]`                | 切片中的任意字符包含在字符串中         |
| `F: FnMut(char) -> bool` | `F` 对字符串中的某个字符返回 `true` |
| `&&str`                  | 是子字符串                   |
| `&String`                | 是子字符串                   |


### Ⅱ. 实现层

- `char`
- `&[char]`
- `&str`
- 闭包 `FnMut(char) -> bool`
- 多字符匹配 `MultiCharEq`
- `CharsEq`
- `StrSearcher` 等

### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
###  1. `Pattern` for `char`（匹配单个字符）

#### **定义**
匹配某一个具体字符。
#### **示例**

```rust
let idx = "Hello".find('e');
assert_eq!(idx, Some(1));  // 'e' 在位置 1
```

---
### 2. `Pattern` for `&[char]`（匹配任意一个字符）

#### **定义**

匹配字符集合中的**任意一个**。
#### **示例**

```rust
let idx = "Hello".find(&['e', 'o']);
assert_eq!(idx, Some(1));  // 'e' 匹配
```

---

###  3. `Pattern` for `[char; N]`（数组，固定长度）

#### **定义**

与 `&[char]` 等价，但支持字面量数组。
#### **示例**

```rust
let idx = "world".find(['a', 'o']); 
assert_eq!(idx, Some(1));  // 'o'
```

---

### ### ### 4. `Pattern` for `&str`（匹配子字符串）

#### **定义**

匹配整个字符串。

#### **示例**

```rust
let idx = "Hello world".find("world");
assert_eq!(idx, Some(6));
```

---

###  5. `Pattern` for `char` predicate（`FnMut(char) -> bool`）

#### **定义**

匹配满足某个布尔条件的字符。

#### **示例**

```rust
let idx = "Hello".find(|c: char| c.is_uppercase());
assert_eq!(idx, Some(0));  // 'H'
```

---

###  6. `Pattern` for `CharsEq`（特殊字符比较器，内部使用）

#### **定义**

用于比较大小写折叠等特殊用途，一般由标准库内部构造。

#### **示例（等价于匹配某个字符）**

```rust
use std::str::pattern::Pattern;

let idx = "abc".find('b'.into());
assert_eq!(idx, Some(1));
```

> 说明：一般用户不会显式使用 `CharsEq`，但可以通过 `find(CharEq)` 方式触发。

---

### 7. `Pattern` for `StrSearcher`（底层搜索器，内部使用）

#### **定义**

标准库内部实现 `find`、`contains`、`strip_prefix` 的搜索引擎。  
你无法直接构造 `StrSearcher`，但可以观察其行为。

#### **示例（等价于子串匹配）**

```rust
let idx = "abcde".find("cd");
assert_eq!(idx, Some(2));
```

---

###  8. `Pattern` for `&String`（自动 deref 为 `&str`）

#### **定义**

Deref 到 `&str`，实际走 `&str` 实现。

#### **示例**

```rust
let pat = "llo".to_string();
let idx = "Hello".find(&pat);
assert_eq!(idx, Some(2));
```

---

###  9. `Pattern` for `&&str`（多层引用，全部自动解引用）

#### **示例**

```rust
let pat = "lo";
let idx = "Hello".find(&&pat);
assert_eq!(idx, Some(3));
```

---

###  10. `Pattern` for `&[u8]`（UTF-8 字符数组 → 字符匹配）

#### **定义**

历史兼容，较少用。匹配 UTF-8 字符。

#### **示例**

```rust
let idx = "abc".find(&[98]); // ASCII 'b'
assert_eq!(idx, Some(1));
```

---

### 11. `Pattern` for `char` slices（匹配任意一个 char）

等价于前面的 `[char]`，这里给一个不同示例。

#### **示例**

```rust
let idx = "rustacean".find(&['u', 's']);
assert_eq!(idx, Some(1)); // 'u'
```

---

###  12. `Pattern` for `Fn(char) -> bool`（非闭包函数）

#### 示例

```rust
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}
let idx = "rust".find(is_vowel);
assert_eq!(idx, Some(2)); // 'u'
```

---

### 总结

Rust 的 `Pattern` trait 支持 **多类匹配方式**：

- **精确字符**：`char`
    
- **字符集合**：`[char]`, `&[char]`
    
- **子串匹配**：`&str`
    
- **谓词匹配**：`FnMut(char)->bool`
    
- **内部优化搜索器**：`StrSearcher` / `CharsEq`
    
- **自动解引用与自动类型转换**：`&String`、`&&str`
    

你可以用 `.find()`, `.split()`, `.strip_prefix()` 等 API 利用这些模式，Rust 会根据类型自动选择搜索策略。

---

### ### ### 学习方法论

- 记住：**Pattern 是一个“模式匹配器” trait，类型不同 → 匹配策略不同**
    
- 多尝试用 `.find()` 传入各种类型，理解隐式 dispatch
    
- 阅读 `pattern.rs` 源码理解标准库如何为每种模式生成搜索器
    
- 对比 C 的 strchr/strstr，理解 Rust 如何多策略统一
    

---

### ### ### 练习题

1. 用 `find()`、`rfind()` 测试每种 `Pattern` 行为。
    
2. 写一个接受 `impl Pattern` 的自定义函数，实现你自己的 `.search()`。
    
3. 使用 mask 优化字符集合匹配并与 `[char]` Pattern 的性能比较。
    
4. 用 SIMD 写你自己的搜索器，并与标准库 `StrSearcher` 对比。
    

---

如果你愿意，我可以给你画一份 **Pattern Dispatch 决策图**，展示每种类型是如何在编译期选择具体实现的。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-num-from_str_int_impl](../../num/Rust-num-from_str_int_impl.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
