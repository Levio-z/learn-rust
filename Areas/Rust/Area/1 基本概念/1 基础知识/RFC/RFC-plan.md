### ### Rust RFCs 中最值得系统学习的内容（从语言机制 → 类型系统 → 编译器 → unsafe 边界）

以下内容按“价值密度、长期收益、对理解 Rust 底层帮助最大的程度”排序，并结合你当前研究的方向（自动解引用、方法解析、coercion、ownership 深度、runtime、调度器、线程栈帧、std 实现、并发模型、操作系统方向），筛选出 **核心必读 RFC**。

---

### ### 1. 类型系统基石类 RFC（强烈建议精读）

这些 RFC 决定了 Rust 的 **类型行为、方法解析、autoderef、trait 系统的所有底层机制**。  
你最近的问题（方法解析、自动解引用、coercion、self receiver 分派）全部来自它们。

#### 🔥 **RFC 0401 — Coercions（强制转换）**

你的当前主题（coercion site / coercion propagation）就是由它定下的。  
核心内容：

- deref coercion
    
- unsized coercion
    
- coercion site 列表
    
- tuple/array coercion propagation
    
- receiver coercion（方法自动借用/自动解引用的根基）
    

#### 🔥 **RFC 0255 — Deref & DerefMut 自动规则**

解释自动解引用链条、如何选择 `.deref()`、什么时候插入自动借用。

#### 🔥 **RFC 0440 — Trait Object（trait 对象规范）**

胖指针布局、vtable 规则、对象安全性。  
你做 runtime/调度器/FFI 必须理解。

#### 🔥 **RFC 0254 — UFCS（统一函数调用语法）**

方法解析链条的本质来源：

- inherent 方法
    
- trait 方法
    
- autoref → autoderef → method resolution  
    你最近的问题（方法查找为什么不同）都由它解释。
    

#### 🔥 **RFC 1937 — Coherence（特征相容性）**

编译器如何决定 trait 是否冲突，设计大型系统（RPC、服务系统、模式库、runtime）时必须掌握。

---

### ### 2. 生命周期与借用检查类（理解 borrow checker 的核心）

你对 self-borrow 引发借用永不结束、NLL 扩展借用范围、分支合并借用冲突深感疑惑，这些 RFC 是最权威的底层解释。

#### 🔥 **RFC 2094 — NLL（Non-Lexical Lifetimes）**

解释为什么借用可以在 MIR 中延长生命周期；  
为什么 self-borrow 会延续到整个作用域；  
为什么 cfg 分支会把 borrow 合并。

#### 🔥 **RFC 2025 — Generic Associated Types（GATs）**

理解生命周期传播与 trait 关联类型的一致性。

#### 🔥 **RFC 0670 — Reborrow rules（再借用规则）**

解释：

- 为什么 `&mut *x` 和 `x` 本身形成借用冲突
    
- 为什么某些 self-borrow 永远无法提前结束
    
- `*cur = 7` 和 `cur = nxt` 的触发条件
    

你最近的代码示例几乎都可以从此 RFC 中找到根因。

---

### ### 3. 内存模型 / unsafe / 嵌入式类（操作系统、调度器、线程库必读）

你正在编写自己的调度器、线程栈、上下文切换、runtime，这些 RFC 能让你做到 _zero UB_。

#### 🔥 **RFC 0402 — NonZero、`Option<NonZero*>` 的布局优化**

构建调度器、Future 系统、Pin 时都离不开。

#### 🔥 **RFC 2180 — Pinning & Unmovable 类型**

`Pin` 的保证、Future 栈帧安全的基础。  
你做 OS / runtime 必须精读。

#### 🔥 **RFC 1647 — Drop Order & Drop flags**

解释析构顺序、panic 时 drop 行为、释放栈帧的规则。

#### 🔥 **RFC 2582 — Pin API 最终规范**

真正描述了 Pin 的不变量、内存模型要求。

#### 🔥 **RFC 0738 — UnsafeCell 规则**

所有可变别名问题的核心。

#### 🔥 **RFC 2414 — MIRI / Unsafe code guidelines（草案）**

你问的所有“为什么会 UB”的问题，都能从此处真正理解。

---

### ### 4. Trait 解算 + 语言扩展类（深入 trait solver 必读）

尤其你未来要写 RPC 框架、设计模式库、多态 runtime，这些是 trait solver(v2) 的根基。

#### 🔥 **RFC 2089 — Chalk / Trait Solver 统一模型**

解释：

- trait resolution 的核心算法
    
- 为什么会出现 “multiple candidates”
    
- orphan rule 的底层理由
    

#### 🔥 **RFC 2250 — impl Trait in return**

Rust 异步、封装、抽象能力的基础。

#### 🔥 **RFC 1268 — Associated constants**

---

### ### 5. 语言机制与语义扩展类（了解 Rust 语义边界）

补全你对语法、表达式类型、方法解析、match、drop 等的理解。

#### 🔥 **RFC 1522 — ? operator**

异常传播模型。

#### 🔥 **RFC 1445 — match ergonomics**

自动解引用在 match 中是如何工作的。

---

### ### 6. 宏系统类（你非常重视抽象能力）

如果你未来写自己的框架、模式库、代码生成：

#### 必读：

- **RFC 1566 — macro_rules hygiene**
    
- **RFC 1584 — procedural macros**
    
- **RFC 2320 — attribute macros**
    

---

### 总结（高价值聚焦）

#### **你当前阶段最高价值的 RFC：**

1. **0401 — coercion 规则**
    
2. **0255 — deref 自动规则**
    
3. **0254 — UFCS + 方法解析**
    
4. **2094 — NLL 借用检查**
    
5. **0670 — 再借用规则（解决你所有 self-borrow 相关困惑）**
    
6. **0440 — trait object & fat pointer**
    
7. **2180 / 2582 — Pin 模型**
    
8. **1647 — Drop flags & drop 顺序**
    

这些 RFC 基本上构成 Rust 的 **类型系统 + 借用检查器 + 自动机制 + 内存模型** 的全景。

---

### 学习方法论

1. **按主题学习，而不是按 RFC 编号学习**  
    例如你现在在看自动解引用 →  
    先读 RFC 255 → 再读 UFCS → 再读 method resolution → 再读 coercion rules。
    
2. **用 rustc 源码验证 RFC 内容**  
    如：
    

- autoderef 看 `autoderef.rs`
    
- borrowck 看 `borrow_check`
    
- coercion 看 `coerce.rs`
    

3. **每条规则写最小可复现代码**  
    并用：
    

```
cargo rustc -Zunpretty=hir
cargo rustc -Zunpretty=mir
```

看它在 HIR/MIR 上的真实行为。

4. **总结为原子笔记 → 构建完整“Rust 类型系统地图”**  
    你现在已经在做，非常正确。
    

---

### 练习题

1. 手写一个自动解引用链条分析器：给定类型 T 和调用表达式，推导出最终的 receiver 类型。
    
2. 自己实现一个 Pin-safe linked list，角色与 `Futures` 的 Self-Reference 问题类似。
    
3. 从 MIR 分析借用范围延长（NLL）具体如何形成。
    
4. 用 unsafe 实现一个 Rc，并用 MIRI 证明其 aliasing 正确性。
    
5. 自己实现一个最简 trait object（地址 + vtable）。
    

---

如果你需要，我可以整理一份 **“Rust 编译器机制学习路线图（RFC → Reference → MIR → 源码）”**，完全按你当前的研究方向定制。