---
tags:
  - permanent
---
## 1. 核心观点  
> let arr = `[value; N];`编译器会 **复制 `value` N 次**，生成数组。复制要求类型实现 `Copy` 特征。

## 2. 背景/出处  
- 来源：https://os.phil-opp.com/zh-CN/allocator-designs/#kuai-da-xiao
- 引文/摘要：  
```rust
const EMPTY: Option<&'static mut ListNode> = None;
let list_heads: [Option<&'static mut ListNode>; BLOCK_SIZES.len()] = [EMPTY; BLOCK_SIZES.len()];
```

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- Rust 的 可变引用 &mut T 默认不实现 Copy，因为可变引用要求 唯一性。两个同时存在的 &mut 会导致数据竞争，因此 Rust 禁止 &mut T 被隐式复制。 
- 为什么用 `const EMPTY` 可以绕过 `[None; N]` 的 `Copy` 限制
	- const EMPTY: Option<&'static mut ListNode> = None;
	- `EMPTY` 是**编译期常量**。
	- Rust 在编译期就知道它的值是 `None`。
	- 编译器可以 **直接生成数组的每个元素都是 None 的静态内存映射**，而不是依赖运行时复制。
	- 编译器不再按位复制 `EMPTY`，而是**在编译期生成每个数组元素为 `None` 的指针值**。
	- 这里不存在运行时复制 `&mut ListNode`，因此**不需要 Copy**。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
