---
title: 链表中这种不断向后的引用链的生命周期
tags:
  - permanent
---
## 1. 核心观点  
> 生命周期可视化

current = current.next.as_mut().unwrap();
右边获取 current.next 的可变借用

左边赋值给 current

**旧 current 生命周期结束(旧current 生命周期结束后可以自由借用其他字段，除了current.next已经被借用了)，新 current 生命周期开始**，任何时候 **链表中只有一个 &mut 借用活着**

```
时间 → 
[old current 可变借用] ---------结束
                       \
                        [next_node 可变借用] ----> current (新)
```

## 2. 背景/出处  
- 来源：https://os.phil-opp.com/zh-CN/allocator-designs/#shi-xian-1
- 引文/摘要：  

```rust
current = current.next.as_mut().unwrap();
```




## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- **字段可变借用是按字段独立的**  


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
