## **1. 使用 `std::any::type_name` 获取类型**


```rust
fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
}

fn main() {
    let a = 'a';      // char
    let b = &'a';     // &char

    print_type_of(&a); // Type: char
    print_type_of(&b); // Type: &char
}
```

Type: char
Type: &char

---

## **2. 模式匹配（`match`）**

```rust
fn check_char_type(c: &dyn std::any::Any) {
    match c.downcast_ref::<char>() {
        Some(_) => println!("It's a char"),
        None => match c.downcast_ref::<&char>() {
            Some(_) => println!("It's a &char"),
            None => println!("Unknown type"),
        },
    }
}

fn main() {
    let a = 'a';
    let b = &'a';

    check_char_type(&a); // It's a char
    check_char_type(&b); // It's a &char
}

```

---

## **3. 使用 `std::mem::size_of` 检查大小**

`char` 在 Rust 中固定占 **4 字节**（Unicode 标量值），而 `&char` 是一个指针（通常占 `usize` 大小，即 8 字节（64 位系统））。

```rust
fn main() {
    let a = 'a';
    let b = &a;

    println!("Size of char: {}", std::mem::size_of_val(&a)); // 4
    println!("Size of &char: {}", std::mem::size_of_val(&b)); // 8 (64-bit)
}
```

输出（64 位系统）：

Size of char: 4
Size of &char: 8