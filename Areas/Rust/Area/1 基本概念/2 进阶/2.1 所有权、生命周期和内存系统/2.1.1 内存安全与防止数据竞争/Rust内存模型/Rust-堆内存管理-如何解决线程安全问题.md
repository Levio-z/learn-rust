---
tags:
  - permanent
---

## 1. 核心观点  
### Ⅰ. 概念层

当线程1的调用栈分配字符串"hello world"时，线程2不能直接访问，因为值被唯一的scope拥有（线程1的scope）。这迫使开发者必须使用Arc（原子引用计数）等智能指针来共享内存，通过引用计数追踪访问权。

### Ⅱ. 应用层





### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
	- [**Variables Live in the Stack**](https://rust-book.cs.brown.edu/ch04-01-what-is-ownership.html#variables-live-in-the-stack)
	- [**Boxes Live in the Heap**](https://rust-book.cs.brown.edu/ch04-01-what-is-ownership.html#boxes-live-in-the-heap)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：
	- [操作系统-内存结构以及生命周期管理](../../../../../../../basic/操作系统/os-note/操作系统-内存结构以及生命周期管理.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
