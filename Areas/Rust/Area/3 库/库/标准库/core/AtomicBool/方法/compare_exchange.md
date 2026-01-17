- 如果当前值与 `current` 相同，则将一个值存储到 [`bool`] 中。
- 返回值是一个结果（`Result`），指示是否写入了新值，并包含先前的值。
- 成功时返回的旧值保证等于 `current`。

`compare_exchange` 接受两个 [`Ordering`] 参数，用来描述此操作的内存排序：
- `success`：描述当比较成功（即当前值等于 `current`）时，读-改-写操作所需的内存排序。
- `failure`：描述当比较失败时，加载操作所需的内存排序。


- 失败排序只能是 `SeqCst`、`Acquire` 或 `Relaxed`
#### 某些平台不支持对 `u8` 做原子操作。此时，Rust 标准库将使用更粗粒度的原子类型（如 `AtomicUsize` 或锁）模拟布尔类型的原子行为。
```rust
let order = match (success, failure) {
    (SeqCst, _) => SeqCst,
    (_, SeqCst) => SeqCst,
    (AcqRel, _) => AcqRel,
    (_, AcqRel) => panic!("there is no such thing as an acquire-release failure ordering"),
    (Release, Acquire) => AcqRel,
    (Acquire, _) => Acquire,
    (_, Acquire) => Acquire,
    (Release, Relaxed) => Release,
    (_, Release) => panic!("there is no such thing as a release failure ordering"),
    (Relaxed, Relaxed) => Relaxed,
};

```
由于模拟实现无法严格区分成功和失败路径，所以统一采用一种最强的 `order` 排序（取两个中的最强者）来作为底层操作的排序。
- 注意这里进行了符合标准的约束检查：
    - `failure` **不能**是 `Release` 或 `AcqRel`（它是纯读取路径，不允许写屏障）；
    - 所以如 `(_, Release)` 和 `(_, AcqRel)` 会 panic。
### 处理模拟逻辑（非原生原子）
```
let old = if current == new {
    self.fetch_or(false, order)
} else {
    self.swap(new, order)
};

```
- 若 `current == new`，理论上什么都不需要做（等价于 no-op），但为了保证顺序语义（如屏障行为），仍调用 `fetch_or(false)`。
- 否则，调用 `swap` 原子替换值。
最后：
```rust
if old == current {
    Ok(old)
} else {
    Err(old)
}

```
表示模拟完成后的结果逻辑：是否真的匹配 `current` 值，决定是否替换成功。

### 原生硬件支持原子操作
```rust
match unsafe {
    atomic_compare_exchange(self.v.get(), current as u8, new as u8, success, failure)
} {
    Ok(x) => Ok(x != 0),
    Err(x) => Err(x != 0),
}

```
这里调用的是平台提供的**原子比较交换指令**，通常会转化为 LLVM 的 `cmpxchg` 指令，在硬件上实现原子性。

- `self.v.get()`：获得 `AtomicBool` 内部的原始指针（`UnsafeCell<u8>`）；
    
- `current`、`new`：布尔值转换成 `u8`（0/1）传给底层原子操作；
    
- `success`、`failure`：直接传入 LLVM 的原子语义。