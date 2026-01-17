### Rust example 2: RefCell (1/4)
**背景（Context）：** 完全禁止共享可变访问（SMA）在某些场景下不现实，例如并发

**解决方案（Solution）：内部可变性（interior mutability）**  
通过安全的 API 封装 SMA，好像不存在 SMA 一样

**示例（Example）：`RefCell<T>`**  
在运行时（而非编译时）检查所有权

- `RefCell<T>::try_borrow()`：尝试不可变借用内部值
    
- `RefCell<T>::try_borrow_mut()`：尝试可变借用内部值
    

文档链接：

- [RefCell 文档](https://doc.rust-lang.org/stable/std/cell/struct.RefCell.html?utm_source=chatgpt.com)
    
- [Rust 官方书籍 — 内部可变性](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html?utm_source=chatgpt.com)

```rust
**

fn f1() -> bool { true }

fn f2() -> bool { !f1() }

  

fn main() {

let mut v1 = 42;

let mut v2 = 666;

  

let p1 = if f1() { &v1 } else { &v2 };

  

if f2() {

let p2 = &mut v1;

*p2 = 37;

println!("p2: {}", *p2);

}

println!("p1: {}", *p1);

}

**
```

https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=c07efb0ed16980ef85d09568382114f9

**假设 `f1()` 和 `f2()` 是复杂且互斥的条件（不是 `f1() && f2()`）**

这是安全的，因为 `p1` 和 `p2` 没有发生别名（alias）

**编译错误原因**：  
类型检查器无法判断安全性  
由于条件过于复杂，出现错误信息：  
“无法将 `v1` 可变借用，因为它同时也被不可变借用。”

### **Rust example 2: RefCell (3/4)**

```rust
**

use std::cell::RefCell;

  

fn f1() -> bool { true }

fn f2() -> bool { !f1() }

  

fn main() {

let v1 = RefCell::new(42);

let v2 = RefCell::new(666);

  

let p1 = if f1() { &v1 } else { &v2 }

.try_borrow().unwrap();

  

if f2() {

let mut p2 = v1  
.try_borrow_mut().unwrap();

*p2 = 37;

println!("p2: {}", *p2);

}

  

println!("p1: {}", *p1);

}

**
```

- **所有权在运行时检查  （`try_borrow()`、`try_borrow_mut()`）**  
	- 编译并按预期执行  
- 输出：“p1: 42”
- 如果 `f1() && f2()` 为真，`try_borrow_mut()` 会在**运行时**失败（而非编译时）  报错信息：“thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value'”

### **Rust example 2: RefCell (4/4)**

**内部可变性（Interior mutability）：** 在非 SMA 类型中封装 SMA
**安全 API：** 表面上没有 SMA

```rust
pub fn try_borrow_mut(&self) -> Result<RefMut<T>, BorrowMutError>
```


（不可变地借用 `self`）
**潜在不安全的实现：** 实际可能存在 SMA
```rust
... unsafe { &mut *self.value.get() }, …
```

[源码链接](https://doc.rust-lang.org/1.63.0/src/core/cell.rs.html?utm_source=chatgpt.com#1732)

**“Unsafe”**：桥接**无 SMA 的 API**和**含 SMA 的实现**
-   需要手动检查：唯一需要进行安全性检查的部分
-   应明确标注 `unsafe`

