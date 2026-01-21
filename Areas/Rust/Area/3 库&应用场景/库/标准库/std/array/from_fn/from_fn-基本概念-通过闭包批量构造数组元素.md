这段代码展示了 Rust **数组初始化的高级方式**：使用 `std::array::from_fn` 通过闭包批量构造数组元素。关键点在于：
- `array: [String; 8]` 是 **长度固定为 8 的 String 数组**
- `std::array::from_fn` 接收一个闭包，闭包入参是元素下标 `_i`
- 闭包每被调用一次，会产生一个新的 `String`
- 因为数组有长度 8，闭包会被调用 8 次，生成 8 个独立的 `String`

最终，`println!("{:#?}", array);` 使用结构化调试格式输出该数组。

### from_fn 的作用与源码机制
#### from_fn 的签名
```rust
pub fn from_fn<F, T, const N: usize>(func: F) -> [T; N]
where
    F: FnMut(usize) -> T,
```
#### **原理要点**
1. **编译器提前知道数组大小 N**
2. 它会分配长度 N 的未初始化数组缓冲区
3. 从 `i = 0` 到 `i = N-1` 调用闭包：
```rust
func(i)
```
4. 每次调用返回一个完整的 `T`
5. 所有元素填充完毕后，返回 `[T; N]`
### 底层模型
```rust
let mut data: [MaybeUninit<T>; N] = uninit_array();

for i in 0..N {
    data[i].write(func(i));
}

unsafe { assume_init(data) }
```
核心优势：**避免必须写重复代码来构造数组元素**，尤其适合非 Copy 类型（如 String）。