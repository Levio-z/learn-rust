
- **功能**：获取 `Option<T>` 中值的不可变引用
- **所有权**：保留原来的所有权（self 不被 move）
- **示例**：
```rust
let x = Some(String::from("hello"));
if let Some(v) = x.as_ref() {
    println!("Length: {}", v.len()); // v: &String
}
```

### 详解
[2 as_ref](../../../2.1%20所有权、生命周期和内存系统/2.1.2%20所有权系统/引用机制/自动解引用机制/2%20as_ref.md)
### 剖析
[as_ref 剖析](../../../2.1%20所有权、生命周期和内存系统/2.1.2%20所有权系统/引用机制/自动解引用机制/as_ref%20剖析.md)