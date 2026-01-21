---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### Rust 中 `for_each` vs `try_for_each` 与 `?` 的区别

#### 核心问题

你写：

```rust
vec.iter().for_each(|row| wtr.serialize(row)?);
```

会报错，但

```rust
vec.iter().try_for_each(|row| wtr.serialize(row))?;
```

可以工作，原因在于 **错误传播机制和闭包返回值类型**。

---

#### 1️⃣ `for_each` 的签名

```rust
fn for_each<F>(self, f: F)
where
    F: FnMut(Self::Item)
```
- 闭包 `f` **必须返回 `()`**
- 闭包里使用 `?` 相当于返回 `Result`，**和要求的 `()` 类型不匹配**
- Rust 编译器报错：`?` 只能在返回 `Result` 的函数/闭包里使用。

简单理解：`for_each` 不知道如何“把 `?` 的错误向外传递”，它只期待闭包返回 `()`。

---

#### 2️⃣ `try_for_each` 的签名

```rust
fn try_for_each<F, R>(self, f: F) -> R
where
    F: FnMut(Self::Item) -> Result<(), E>,
    R: From<Result<(), E>>,
```
- 闭包 `f` **必须返回 `Result`**
- `try_for_each` 会**自动把第一个 `Err` 返回**给调用者
- 所以闭包里的 `?` 可以正常使用，错误会被向外传播
等价逻辑：
```rust
for item in iter {
    f(item)?; // 第一个 Err 会立即返回
}
```

---

#### 3️⃣ 对比理解

|方法|闭包返回值|支持 `?` 吗|作用|
|---|---|---|---|
|`for_each`|`()`|❌|只做副作用，不处理 Result|
|`try_for_each`|`Result`|✅|遇到 Err 会提前返回，支持错误传播|

> 💡 本质上是 **`?` 只能在返回 `Result` 的函数/闭包里使用**，`for_each` 的闭包返回 `()`，所以不行。

---

#### 4️⃣ 代码示例

```rust
vec.iter().try_for_each(|row| wtr.serialize(row))?;
```

- `wtr.serialize(row)` 返回 `csv::Result<()>`
    
- `?` 会把错误传播到外层 `main() -> csv::Result<()>`
    
- 遍历完所有行或遇到第一个错误即停止
    

---

总结：

- `for_each` 是“盲目执行”，不能传播错误
    
- `try_for_each` 是“可中断执行”，支持 `?` 错误传播
    

---

如果你想，我可以画一个**流程图对比 for_each 和 try_for_each 错误传播的区别**，视觉上很直观。你希望我画吗？ 

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
