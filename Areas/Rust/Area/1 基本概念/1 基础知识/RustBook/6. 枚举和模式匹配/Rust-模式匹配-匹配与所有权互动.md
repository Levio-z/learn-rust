---
tags:
  - permanent
---
## 1. 核心观点  

- 默认匹配值会移动数据
- 匹配引用会下推到内部引用，详情见[Rust-模式-绑定方式-基本概念](Rust-模式-绑定方式-基本概念.md)

## 2. 背景/出处  
- 来源：https://rust-book.cs.brown.edu/ch06-02-match.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 不忽略匹配值会移动数据
如果枚举包含不可复制的数据（如 String），则应注意**匹配项是否会移动或借用该数据**。例如，此程序使用 `Option<String>` 将编译：
```rust
 fn main() {
let opt: Option<String> = 
    Some(String::from("Hello world"));

match opt {
    Some(_) => println!("Some!"),
    None => println!("None!")
};

println!("{:?}", opt);
 }
```

但是，如果我们将 `Some（_）` 中的占位符替换为变量名称，例如 `Some（s），` 那么程序将不会编译：
```rust
fn main() {
let opt: Option<String> = 
    Some(String::from("Hello world"));

match opt {
    // _ became s
    Some(s) => println!("Some: {}", s),
    None => println!("None!")
};

println!("{:?}", opt);
}
```

`opt` 是一个普通枚举 — 它的类型是 `Option<String>` 而不是像 `&Option<String>` 这样的引用。因此，**`opt` 上的匹配将移动非忽略的字段，例如 `s`**。请注意，与第一个程序相比，`opt` 在第二个程序中更快地失去读取和拥有权限。在匹配表达式之后，`opt` 中的数据已被移动，因此在 `println` 中读取 `opt` 是非法的。
### 如果我们想在不移动其内容的情况下查看 `opt`，惯用的解决方案是匹配引用：
```rust
fn main() {
let opt: Option<String> = 
    Some(String::from("Hello world"));

// opt became &opt
match &opt {
    Some(s) => println!("Some: {}", s),
    None => println!("None!")
};

println!("{:?}", opt);
}
```

Rust 会将引用从外部枚举 `&Option<String>` “下推”到内部字段 `&String`。因此 `s` 的类型为 `&String`，匹配后可以使用 `opt`。为了更好地理解这种“下推”机制，请参阅 Rust 参考中有关[绑定模式](https://doc.rust-lang.org/reference/patterns.html#binding-modes)的部分。


## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-模式匹配-基本概念](Rust-模式匹配-基本概念.md)
- 后续卡片：
	- [Rust-模式-绑定方式-基本概念](Rust-模式-绑定方式-基本概念.md)
	- [Rust-模式-模式分类-Non-reference patterns非引用模式](Rust-模式-模式分类-Non-reference%20patterns非引用模式.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

