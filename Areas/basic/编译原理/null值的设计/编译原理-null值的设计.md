---
tags:
  - permanent
---
## 1. 核心观点  

Tony Hoare 所说的“十亿美元错误”本质上是 **空指针引用（null reference）** 的设计缺陷：

- **空值容易被误用**：如果程序试图将 null 当作有效值使用，就会导致运行时错误、漏洞或崩溃。
    
- **不可见性**：在很多语言中，null 是隐式存在的，编译器无法在类型层面强制检查，从而使错误容易发生。
    
- **概念 vs 实现**：null 本身是一个有用概念——表示“当前无值”或“缺失值”。问题出在它没有安全的实现方式。

## 2. 背景/出处  

在他 2009 年的演讲“Null References： The Billion Dollar Mistake”中，null 的发明者 Tony Hoare 是这样说的：

> I call it my billion-dollar mistake. At that time, I was designing the first comprehensive type system for references in an object-oriented language. My goal was to ensure that all use of references should be absolutely safe, with checking performed automatically by the compiler. But I couldn’t resist the temptation to put in a null reference, simply because it was so easy to implement. This has led to innumerable errors, vulnerabilities, and system crashes, which have probably caused a billion dollars of pain and damage in the last forty years.  
> 我称之为我价值数十亿美元的错误。当时，我正在设计第一个使用面向对象语言的引用的综合类型系统。我的目标是确保所有引用的使用都应该绝对安全，并由编译器自动执行检查。但我无法抗拒放入空引用的诱惑，仅仅因为它很容易实现。这导致了无数的错误、漏洞和系统崩溃，在过去四十年中可能造成了十亿美元的痛苦和损害。

## 3. 展开说明  

## 4. 与其他卡片的关联  
- 前置卡片：
	- 
- 后续卡片：
	- [Rust-Option vs 空值-基本概念](../../../Rust/Area/1%20基本概念/1%20基础知识/RustBook/6.%20枚举和模式匹配/Rust-Option%20vs%20空值-基本概念.md)
- 相似主题：
- 

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
