---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层

字符串切片：&str类型的视图，如let s_slice: &str = &s，只包含指向字符串缓冲区的指针。

- 切片`as[1]`取元素，会直接move

### Ⅱ. 实现层

### 常用方法
[ Rust 中 `get(..5)` 用法](#%20Rust%20中%20`get(..5)`%20用法)

### Ⅲ. 原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### Rust 中 `get(..5)` 用法

### 1. 方法签名

切片 `[T]` 和数组 `[T; N]` 的 `get` 方法不仅可以传单个索引，还可以传 **范围（Range）**：

```rust
impl<T> [T] {
    pub fn get<R>(&self, range: R) -> Option<&[T]>
    where
        R: SliceIndex<[T]>;
}
```

- `R` 可以是 `RangeFull`, `Range`, `RangeFrom`, `RangeTo`, `RangeInclusive` 等
    
- 返回 `Option<&[T]>` 或 `Option<&mut [T]>`
    

---

### 2. `..5` 的含义

```rust
let slice = &[10, 20, 30, 40, 50];
let part = slice.get(..5);
```

- `..5` 等价于 `0..5`
    
- 表示从 **起始索引 0 到索引 5（不包含 5）**
    
- 返回 `Some(&[10, 20, 30, 40, 50])`
    

如果超出范围：

```rust
let slice = &[10, 20, 30];
let part = slice.get(..5);
assert!(part.is_none()); // 越界返回 None
```

---

### 3. 可变切片 `get_mut` 也支持

```rust
let mut arr = [1, 2, 3, 4, 5];
if let Some(sub) = arr.get_mut(..3) {
    sub[0] += 10;
}
println!("{:?}", arr); // [11, 2, 3, 4, 5]
```

- 返回 `&mut [T]`，可安全修改子切片
- 越界仍然返回 `None`
    

---

### 4. 总结

|范围类型|示例|结果类型|
|---|---|---|
|`..`|全部|`Option<&[T]>`|
|`..end`|前 end 个元素|`Option<&[T]>`|
|`start..end`|start 到 end-1|`Option<&[T]>`|
|`start..`|start 到末尾|`Option<&[T]>`|
|`..=end`|前 end+1 个元素|`Option<&[T]>`|

特点：
- 安全：越界返回 `None`
- 支持不可变和可变引用
- 数组会自动解引用为切片
    



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	-    Methods
		- [Rust-slice 切片-get(..)](methods/Rust-slice%20切片-get(..).md)
		- [Rust-slice-范围切片](../../../../../../../../Projects/Rust/1brc/4%20note/note/inbox/Rust-slice-范围切片.md)
		- [Rust-slice-as_chunks](../../../../../../../../Projects/Rust/1brc/4%20note/note/inbox/Rust-slice-as_chunks.md)
- 因此数组的 `get` 方法与切片一致：
	- `let arr: [i32; 3] = [1, 2, 3]; let x: Option<&i32> = arr.get(1);`
	- 数组也有 `get_mut` 可获得可变引用

- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
