---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

避免为不同数据类型重复编写相同算法的代码，符合DRY原则
- rust实现：[Rust-trait-基本概念-TOC](../2.2.3%20特型/Rust-trait-基本概念-TOC.md)
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
- Rust实现:
    - 使用两个泛型参数B和F
    - F参数有trait约束：FnMut(Iter) -> B
    - 包含where子句对Self和F进行约束
- JavaScript实现:
    - 参数类型无约束
    - 运行时才能发现类型错误
- TypeScript改进:
    - 添加类型系统
    - 引入泛型避免重复代码
```rust
fn map<B, F>(self, f: F) -> Map<Self, F>
where
    Self: Sized,
    F: FnMut(Self::Item) -> B,
{
    // 函数体实现
}

```
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
