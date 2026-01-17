**Part 4: implementation**
#### **实现来源**

GitHub 地址：[CS431 Lock 实现](https://github.com/kaist-cp/cs431/tree/main/src/lock?utm_source=chatgpt.com)

包括：

- **Spinlock**：自旋锁
    
- **Ticket lock**：票锁
    
- **CLH lock**：链式自旋锁
    
- **MCS lock**：队列锁
    
- **MCS parking lock**：带停车机制的 MCS 锁
    

---

#### **`RawLock` trait**

- **定义与作用**：  
    定义“原始锁”（raw lock）的接口规范。
    
- **API**：
    
    - `lock()`：获取锁
        
    - `unlock()`：释放锁
        
    - `Token`：表示锁持有者状态
        
    - `Default`、`Send`、`Sync`：支持默认构造及跨线程安全
        
- **保证**：同一时刻只有一个执行单元可以获取锁
    
- **实现者**：Spinlock、Ticket lock、CLH lock 等
    

---

#### **各种锁的权衡**

- **Simple（简单）**：实现容易，易于理解
    
- **Fast (uncontended)**：在无竞争时速度快
    
- **Compact（紧凑）**：占用内存少
    
- **Scalable（可扩展）**：适合多线程高并发
    
- **Fair（公平）**：按顺序获取锁，避免饥饿
    
- **Energy-efficient（节能）**：降低 CPU 空转和功耗
    

不同锁设计在性能、复杂度和能耗上存在折中，需要根据具体场景选择。


### Spinlock 实现解析（忽略内存序 acquire/release）

---

#### **结构定义**

```rust
pub struct RawSpinLock {
    inner: AtomicBool, // true 表示锁已被占用，false 表示锁空闲
}
```

-   **RawSpinLock**：自旋锁的原始实现
    
-   **inner**：使用 `AtomicBool` 作为锁状态标记
    
    -   `true` → 已锁定
        
    -   `false` → 未锁定
        

---

#### **锁获取**

```rust
pub fn lock(&self) {
    while self.inner.compare_and_swap(false, true).is_err() {} // RMW 操作
}
```

-   循环尝试将 `inner` 从 `false` 改为 `true`
    
-   **compare\_and\_swap**：原子读-改-写（RMW）操作
    
-   当锁被占用时，会持续自旋（busy-wait）直到获取成功
    

---

#### **锁释放**

```rust
pub fn unlock(&self) {
    self.inner.store(false); // 非 RMW 操作，因为锁是独占的
}
```

-   将 `inner` 设置为 `false`
    
-   由于锁是独占的，没有竞争冲突，不需要原子读-改-写
    

---

### 自旋锁带内存序的实现翻译

---

#### **锁获取**

```rust
pub fn lock(&self) {
    while self.inner.cas(false, true, acquire).is_err() {}
}
```

-   使用 **`compare_and_swap`（CAS）** 尝试将 `inner` 从 `false` 变为 `true`
    
-   **`acquire` 内存序**：保证在获取锁之后，对共享数据的读写不会被 CPU/编译器重排序
    
-   如果锁已经被其他线程持有，会 **持续自旋（busy-wait）** 直到获取成功
    

---

#### **锁释放**

```rust
pub fn unlock(&self) {
    self.inner.store(false, release);
}
```

-   将 `inner` 设置为 `false`
    
-   **`release` 内存序**：保证在释放锁之前，对共享数据的写入已经对其他线程可见
    

---

#### **行为保证**

-   **锁独占性**：一次只能有一个线程持有该锁
    
-   如果锁已经被占用，调用 `lock()` 的线程会一直自旋等待
    ![](../6%20rust/asserts/Pasted%20image%2020250905210946.png)


### The key ideas of the other locks

#### **通过 CAS 保证互斥**

- **互斥保证**：使用 CAS（Compare-And-Swap）原子操作来确保同一时刻只有一个线程进入临界区。
    
- **顺序保证**：从一个临界区的结束到另一个临界区的开始，确保访问顺序正确。
    
- **位置标记**：
    
    - **Ticket lock** 使用 `curr` 指针标记当前服务的票号
        
    - **CLH / MCS lock** 为每个等待线程分配一个新的节点（位置）用于排队
        

---

#### **通过排队和不同位置保证公平性**

- **公平性保证**：通过在不同位置等待并按顺序访问锁
    
- **顺序操作**：使用公平的原子指令（例如 `swap`、`fetch-and-add`）确保线程按顺序进入临界区
    
- **具体实现**：
    
    - **Ticket lock**：
        
        - 排队顺序通过 `next` 指针维护
            
        - 等待顺序通过 `curr` 指针控制
            
    - **CLH / MCS lock**：
        
        - 排队顺序通过 `tail` 指针维护
            
        - 每个等待线程在自己的节点（新位置）上等待
            

---

#### **作业**

- **任务**：分析并推理各种锁实现的正确性（mutual exclusion、顺序性与公平性）
#### **Ticket Lock（票锁）**

- **公平性保证**：通过票号队列保证线程按顺序获取锁
    
- **实现机制**：锁顺序由提前的公平原子操作（`fetch-add` 或 `swap`）决定
    
- **缺点**：API 相对复杂，需要返回票号
    

---

#### **CLH Lock（链式自旋锁）**

- **可扩展性优化**：每个临界区使用独立的自旋位置，减少多线程竞争
    
- **实现机制**：形成自旋位置的队列
    
- **缺点**：空间开销为 O(n)，n 为临界区数量
    

---

#### **MCS Lock（队列锁）**

- **NUMA 感知优化**：线程在自己分配的位置自旋，减少跨节点缓存访问
    
- **缺点**：解锁时可能需要额外的 CAS 操作
    

---

#### **MCS Parking Lock（带停车机制的 MCS 锁）**

- **能耗优化**：线程停车（阻塞）而不是自旋，降低 CPU 空转
    
- **缺点**：在中等竞争场景下性能可能下降

**[A paper on performance evaluation](https://people.csail.mit.edu/nickolai/papers/boyd-wickizer-locks.pdf)**


### 锁问题
