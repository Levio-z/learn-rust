---
tags:
  - note
  - toc
---

在有可能出现错误时，Rust 函数的返回值可以属于一种特殊的类型，该类型可以涵盖两种情况：要么函数正常退出，则函数返回正常的返回值；要么函数执行过程中出错，则函数返回出错的类型。

**Rust 的类型系统保证这种返回值不会在程序员无意识的情况下被滥用，即程序员必须显式对其进行分支判断或者强制排除出错的情况**。**如果不进行任何处理，那么无法从中得到有意义的结果供后续使用或是无法通过编译**。这样，就杜绝了很大一部分因程序员的疏忽产生的错误（如**不加判断地使用某函数返回的空指针**）。

在 Rust 中有两种这样的特殊类型，它们都属于枚举结构：

- `Option<T>` 既可以有值 `Option::Some<T>` ，也有可能没有值 `Option::None`；
    
- `Result<T, E>` 既可以保存某个操作的返回值 `Result::Ok<T>` ，也可以表明操作过程中出现了错误 `Result::Err<E>` 。
    

我们可以使用 `Option/Result` 来保存一个不能确定存在/不存在或是成功/失败的值。之后可以通过匹配 `if let` 或是在能够确定 的场合直接通过 `unwrap` 将里面的值取出。详细的内容可以参考 Rust 官方文档 [3](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter1/6print-and-shutdown-based-on-sbi.html#recoverable-errors) 。
## 闭包

| 方法                                          | 作用                     | 适用类型               | 特点              |
| ------------------------------------------- | ---------------------- | ------------------ | --------------- |
| [`and_then`](notes/`and_then`.md)/`or_else` | 链式，返回Option的情况         | `Result`, `Option` | 闭包返回相同容器类型      |
| `map/map_err`                               | 链式，映射改变值               | `Result`, `Option` | 闭包返回值           |
| `filter`                                    | 链式，条件过滤                | `Option`           | 仅保留满足条件的 `Some` |
| `ok_or`/`ok_or_else`                        | 链式，`Option` → `Result` | `Option`           | 添加错误上下文         |
| `ok`                                        | 链式，`Result → Option`   | `Result`           |                 |
| `map_or`/`map_or_else`                      | 解包，映射并提供默认值            | `Result`, `Option` | 直接处理成功和失败       |
| `unwrap_or`/`unwrap_or_else`                | 解包并提供默认值               | `Result`, `Option` | 直接消费容器          |
| `match`                                     | 完全控制流处理                | 所有类型               | 最灵活，但代码较长       |
## 场景积累
### map_err
### 将一个函数中返回的错误转换为同一类型
```rust
pub fn sum_integers_from_file(file_path: &str) -> Result<i32, io::Error> {
    let f1 = File::open(file_path)?;
    let reader = BufReader::new(f1);
    let mut count = 0;
    for line in reader.lines(){
        let num = line?
            .parse::<i32>()
            .map_err(|_| {
                io::Error::new(ErrorKind::InvalidData, "Invalid number")
            })?;
        count +=num;
    }
    Ok(count)
  
}
```
## ok
### 不关心错误的情况下将 Result 转换为 Option
- 从而简化错误处理，特别是当你只关注成功结果时。
