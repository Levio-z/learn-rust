---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

- **核心方式**：使用 `impl Trait` 或泛型 + Trait Bound 实现静态分派
- **作用**：编译时确定具体调用的 Trait 方法（性能更高）

```rust
trait Animal {
    fn name(&self) -> &'static str;
}
// 静态分派：编译时为每个实现 Animal 的类型生成专属函数
fn animal_name(impl Animal) -> &'static str {
    animal.name()
}
```

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
