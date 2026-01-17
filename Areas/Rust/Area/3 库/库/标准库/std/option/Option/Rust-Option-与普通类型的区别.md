## 1. 核心观点  

Rust 不允许直接将 Option<`T> 与 T 混合使用，因为 Option<T> 可能是 None，即不存在值。
在使用前，必须显式处理 None 情况

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### Option<`T> 与普通类型的区别

`Option<T>` 与 `T` 是两种完全不同的类型，不能直接参与运算或赋值。


```
let x: i8 = 5;
let y: Option<i8> = Some(5);
let sum = x + y; // ❌ 类型不匹配
```
Rust 不允许直接将 Option<`T> 与 T 混合使用，因为 Option<T> 可能是 None，即不存在值。
在使用前，必须显式处理 None 情况，例如使用：

- 模式匹配（match）
- if let Some(v) 解构
- unwrap / unwrap_or 等安全解包函数

## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-Option-基本概念](Rust-Option-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

