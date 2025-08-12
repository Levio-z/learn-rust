

在 Rust 中，**所有泛型参数默认都隐含 `Sized` 约束**。

换句话说，下面这两段代码：

rust

复制编辑

`fn foo<T>(arg: T) { /* ... */ }`

和

rust

复制编辑

`fn foo<T: Sized>(arg: T) { /* ... */ }`

它们 **效果完全一样**，因为 `T` 默认就要求是 `Sized`。

---

### 📌 什么是 `Sized`？

`Sized` 是一个编译期 trait，表示：

> 这个类型在编译时就能确定具体大小。

比如：  
✅ `i32`、`f64`、`String`、`Vec<T>` → 都是 `Sized`  
❌ `str`、`[T]`、`dyn Trait` → 都是 **`?Sized`**（动态大小类型，DST）

---

### 📦 为什么要显式 `?Sized`

当你写：

`fn foo<T: ?Sized>(arg: &T) { /* ... */ }`

你其实是在对编译器说：

> **允许 T 是动态大小类型**，比如 `str`、`dyn Trait`。  
> 注意这时候只能用 `&T`（或者 `Box<T>`），因为裸 `T` 是没法放到栈上的——大小不确定。

---

### 🔍 回到 `HashMap`

标准库里的 `HashMap::get`：

如果不写 `?Sized`：

这会直接 **禁止**：

- `&str` 查询 `HashMap<String, V>`
- `&[u8]` 查询 `HashMap<Vec<u8>, V>`
- `&dyn Trait` 查询 `HashMap<Box<dyn Trait>, V>`

因为 `str`、`[u8]`、`dyn Trait` 都是 `?Sized`。

---

### 总结

默认：泛型参数带 `Sized` 限制。  
显式 `?Sized`：允许动态大小类型。  
`HashMap::get` 明确加 `?Sized`，是为了泛化能力，支持 slice、trait object 等复杂查询。
