---
tags:
  - permanent
---
## 1. 核心观点  

同时忽略多个字段




## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/patterns.html#rest-patterns
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简单示例
在结构模式上，字段按名称、索引（在元组结构的情况下）引用或通过使用 `..`：

```rust
match s {
    Point {x: 10, y: 20} => (),
    Point {y: 10, x: 20} => (),    // order doesn't matter
    Point {x: 10, ..} => (),
    Point {..} => (),
}

match t {
    PointTuple {0: 10, 1: 20} => (),
    PointTuple {1: 10, 0: 20} => (),   // order doesn't matter
    PointTuple {0: 10, ..} => (),
    PointTuple {..} => (),
}

match m {
    Message::Quit => (),
    Message::Move {x: 10, y: 20} => (),
    Message::Move {..} => (),
}
```
如果不使用 `..`，则需要用于匹配结构的结构模式来指定
```rust
match struct_value {
    Struct{a: 10, b: 'X', c: false} => (),
    Struct{a: 10, b: 'X', ref c} => (),
    Struct{a: 10, b: 'X', ref mut c} => (),
    Struct{a: 10, b: 'X', c: _} => (),
    Struct{a: _, b: _, c: _} => (),
}
```




## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-模式-基本概念](Rust-模式-基本概念.md)
	- [Rust-模式-模式分类-StructPatternEtCetera](Rust-模式-模式分类-StructPatternEtCetera.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 分类的原子笔记
- [ ] 实现的原子笔记
