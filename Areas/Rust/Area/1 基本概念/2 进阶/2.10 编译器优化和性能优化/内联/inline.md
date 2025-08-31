### 1. `#[inline]` 的作用到底是什么？

Rust 的 `#[inline]` 属性告诉编译器：

> “这个函数建议在调用处 _展开（inline）其定义_，而不是调用跳转。”

并不是强制内联，而是：

- **给予编译器优化提示**；
    
- **跨 crate 编译时，允许编译器看到函数体，从而可能内联**（特别关键）。
### 2. 为什么 `is_completed` 这种方法适合内联？
短小函数 + 热路径：
```rust

#[inline]
pub fn is_completed(&self) -> bool {
    self.state_and_queued.load(Acquire) == COMPLETE
}

```
- **极短函数**：只做了一个原子读取 + 等值判断；
- **调用频率高**：在 `Once` 或 `Lazy` 等同步原语中经常被调用；
- **作为 fast-path 快速返回**：很多路径会先判断是否已完成，避免锁争用和 futex。
优化空间：
- 避免函数调用开销（特别是跨 crate）；
- 允许编译器做更激进的代码合并和指令流水优化；
- 对于内联的 `Acquire` 语义读取，可以 **与之后的使用合并编排**，更高效。
### 3. 内联在跨 crate 作用更大
如果 `Once` 实现在 **另一个 crate（如 std）** 中，而你在自己项目中调用它：
- 没有 `#[inline]`：编译器只能看到函数签名，无法展开；
- 加了 `#[inline]`：编译器能看到函数体，有机会内联并跨函数进行优化；
这是 Rust `std` 和 `core` 中大量函数都有 `#[inline]` 的原因之一。
### 4. 什么时候不要滥用 `#[inline]`
- 如果函数体很大，内联反而会造成代码膨胀；
- 如果不在热路径上，内联的收益较低；
- 滥用会增加编译时间，减少指令缓存命中率
### 5.分析

|动机|原因|
|---|---|
|✅ 减少函数调用开销|函数非常短小|
|✅ 支持跨 crate 优化|被频繁调用于 `std::sync` 类型|
|✅ 利于 CPU 流水优化|与 fast path 判断结合更紧密|
|✅ 保证 acquire 可见性后续紧跟使用|避免失序优化干扰|
##### 1. 减少函数调用开销 —— 函数非常短小
#### 解释：
- 普通函数调用 = 参数压栈 + 跳转 + 栈帧构造 + 返回（尤其是跨模块/跨 crate 时）。
- 对于像 `is_completed()` 这样只执行一条原子读取语句的小函数，**调用开销往往大于函数体本身的成本**。
如果不内联：
```rust
call is_completed
```
内联后：
```rust
mov rax, [self]    ; Acquire load
cmp rax, COMPLETE
```
变成连续、局部的指令，无函数开销，提高效率。
### 2. 支持跨 crate 优化 —— 被频繁调用于 std::sync 类型
解释：
- Rust 的编译器 **默认不把其它 crate 的函数体暴露出来**（类似黑箱，只看到签名），这样无法内联优化。
- 除非你使用 `#[inline]`，才会在编译时把函数体“公开”给使用它的 crate。
对于像 `Once::is_completed()` 这种：
是 `std::sync::Once` 的组成部分；
- **几乎在所有懒加载、单例初始化中都会用到**（如 `Lazy<T>`、`OnceCell<T>` 等）；
- 频繁出现在高性能路径中。
所以 `#[inline]` 让它可以在跨 crate 的使用中被内联，是性能优化的关键。
#### 3.利于 CPU 流水优化 —— 与 fast path 判断结合更紧密
#### 解释：
- 现代 CPU 执行是“乱序流水线”模式，会预测执行、并行发射、延迟跳转等；
- 如果 `is_completed()` 是函数调用，CPU 很难提前预取和优化执行路径；
- **但如果它内联了，就变成普通条件判断语句**，可直接在流水线中优化；
#### 举例对比：
```rust
// 内联前
if once.is_completed() {
    return cached;
}

// 内联后等价于：
if once.state_and_queued.load(Ordering::Acquire) == COMPLETE {
    return cached;
}

```

```rust
// 内联前
if once.is_completed() {
    return cached;
}

// 内联后等价于：
if once.state_and_queued.load(Ordering::Acquire) == COMPLETE {
    return cached;
}


```
这样编译器就能把它融合到更大的条件判断中，**配合 fast path（快速路径）逻辑做更强指令融合和分支预测优化**。
#### 4. 保证 Acquire 可见性后续紧跟使用 —— 避免失序优化干扰
```rust
if self.state_and_queued.load(Ordering::Acquire) == COMPLETE {
    // ✅ 此处依赖 load 的结果进行逻辑分支
}


```
- `Acquire` 的语义是：**当前线程必须“看到”所有写入完整体之后的状态变化**；
- 如果该读取是在另一个函数（如 `is_completed()`）中调用，而 **编译器没有内联**，就不能保证后续代码会在 acquire 之后。

举个更极端的反例：
```rust
// 未内联版本（更难做编译器全局可见性保证）
if is_completed() {
    do_something(); // 依赖于 acquire 加载可见性
}

```
- 内联后能保证 acquire 的语义立刻作用于下面的语句；
- 未内联时可能失去这种数据依赖链的明显性，**CPU 或编译器可能做出不安全重排（尤其在 unsafe 场景下）**。

所以内联也有利于 **“同步语义的保守传递”**。