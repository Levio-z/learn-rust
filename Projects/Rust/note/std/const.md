### **1. 核心特性**

| 特性           | 说明                                   |
| ------------ | ------------------------------------ |
| **编译时确定**    | 值必须在编译时已知，不能是运行时计算的表达式（除非是简单的常量表达式）。 |
| **全局可用**     | 可以在任何作用域定义（包括全局作用域），且生命周期为整个程序。      |
| **无内存占用**    | 编译器会直接内联（inline）替换常量，不会在运行时分配内存。     |
| **类型必须显式标注** | 必须明确指定类型（如 `const N: i32 = 42;`）。    |
- **命名规范**：通常使用全大写字母和下划线（如 `MAX_SIZE`）。
```rust
const MAX_BUFFER_SIZE: usize = 1024 * 8;  // 编译时可计算的表达式
const PI: f64 = 3.141592653589793;
const GREETING: &str = "Hello, Rust!";
```