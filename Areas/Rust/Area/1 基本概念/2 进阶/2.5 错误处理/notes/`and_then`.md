---
tags:
  - note
---

### Rust `and_then` 详解（含 Playground 示例）

---

#### 1️⃣ 方法签名

```rust
impl<T> Option<T> {
    pub fn and_then<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> Option<U>;
}
```

- `self`：当前 `Option<T>`。
- `f`：闭包，接收 `T` 返回 `Option<U>`。
- 返回值：`Option<U>`。

**内部逻辑 (简化)
```rust
match self {
	Some(val) => f(val),
   // 直接返回闭包的结果，不进行二次包装！ 
   None => None, 
}
```

**关键点**
1. **闭包返回类型**：`Option<U>` (一个已装箱的值)。 
2. **扁平化**：`and_then` 直接返回闭包的结果，避免了 `Option<Option<U>>` 的嵌套。如果闭包返回 `None`，则链式操作中断。

---

#### 2️⃣ 基本用法

```rust
fn main() {
    let opt1 = Some(5);

    // 将 Some(5) 乘以 2，返回 Option
    let result = opt1.and_then(|x| Some(x * 2));
    println!("{:?}", result); // Some(10)

    let opt2: Option<i32> = None;

    let result2 = opt2.and_then(|x| Some(x * 2));
    println!("{:?}", result2); // None
}
```

✅ 解释：

- `Some(5)` 调用闭包，返回 `Some(10)`。
    
- `None` 不调用闭包，直接返回 `None`。
    

---

#### 3️⃣ 结合条件判断

```rust
fn main() {
    let opt = Some(4);

    let result = opt.and_then(|x| {
        if x % 2 == 0 {
            Some(x / 2)
        } else {
            None
        }
    });

    println!("{:?}", result); // Some(2)
}
```

- `and_then` 可以在闭包中加入逻辑，如果条件不满足就返回 `None`。
    

---

#### 4️⃣ 链式使用

```rust
fn main() {
    let opt = Some(8);

    let result = opt
        .and_then(|x| Some(x / 2))
        .and_then(|y| if y > 2 { Some(y * 3) } else { None });

    println!("{:?}", result); // Some(12)
}
```

- 每次 `and_then` 都返回 `Option`，形成安全的链式调用。
    
- 一旦中间返回 `None`，后续闭包不再执行。
    

---

#### 5️⃣ 链表递归应用示意

```rust
impl Solution {
    pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        head.and_then(|mut n| {
            match n.next {
                None => Some(n),
                Some(mut m) => {
                    n.next = Self::swap_pairs(m.next);
                    m.next = Some(n);
                    Some(m)
                }
            }
        })
    }
}
```

- `head` 为 `None` 时递归终止。
    
- 否则递归处理剩余链表，并交换当前节点对。
    
- 利用 `and_then` 避免多层 `match`。
    

---

#### 6️⃣ 在线 Playground

你可以在 Rust Playground 直接运行下面代码体验 `and_then`：
https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=bc094ea9732b40a436c67fe54770ebfc

```rust
fn main() {
    let opt = Some(10);

    let result = opt
        .and_then(|x| Some(x + 5))
        .and_then(|y| if y > 10 { Some(y * 2) } else { None });

    println!("{:?}", result); // Some(30)
}
```

---

#### 小结
- `and_then` 是 **Option / Result 的链式处理方法**。
- 当值存在 (`Some`) 时执行闭包，返回新的 `Option`；值不存在 (`None`) 时直接跳过。
- 非常适合**链表递归**、函数组合、条件链式处理。
---

我可以帮你画一张 **链表递归 + and_then 流程图**，直观展示每次节点如何传递和交换。  
你希望我画吗？