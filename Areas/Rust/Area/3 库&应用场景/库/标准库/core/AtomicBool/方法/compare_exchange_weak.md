###  方法功能对比简表

|方法|行为|
|---|---|
|`compare_exchange`|**强比较交换**：只有当值不相等时才失败，**不会伪失败**|
|`compare_exchange_weak`|**弱比较交换**：除了值不等外，**可能因为 CPU 优化等原因伪失败**|
在某些 **硬件架构** 上（如 ARM、PowerPC），原子操作的实现会使用低开销的 `LL/SC`（Load-Linked / Store-Conditional）指令对。
- 这类指令受干扰容易失败（即“伪失败”），但性能高；
- `compare_exchange_weak` 允许这种失败，**便于底层用这种方式实现**；
- `compare_exchange` 则要求“失败必须是值不等”，因此要求更强（有些平台需要额外机制保证），开销可能更大。

### 举个 LL/SC 模拟 `compare_exchange` 的过程
```
atomic.compare_exchange_weak(current, new, ...);
```
底层会是：
```rust
LL:   t1 = load_linked(addr)     // 读取 + 监听该地址
      if t1 != current:
          return Err(t1)
SC:   ok = store_conditional(addr, new)
      if !ok:
          return Err(current) // 👈 注意：这里失败时值没变！但仍然失败（伪失败）
      else:
          return Ok(current)


```
1. LL (load-linked): 读取内存值，同时在 CPU 内部“监听”这个地址
2. 做逻辑判断
3. SC (store-conditional): 只有在“监听没有被打断”的情况下才允许写入
如果在 LL 和 SC 之间，有任何事情打扰了这块内存，比如：
- **其他 CPU 核访问了这块地址（即使只是读！）**
- 发生了上下文切换
- CPU 内部缓存失效或其他微架构事件
#### 为什么允许这种设计？

**性能换正确性**：
- LL/SC 是无锁、无总线锁定的机制，非常轻量、高效；
- 为了使用它，我们必须接受“写可能失败”，即 **compare_exchange_weak 可能失败，即使值是对的**；
- 所以你得写个循环来重试（编译器可自动生成）。
