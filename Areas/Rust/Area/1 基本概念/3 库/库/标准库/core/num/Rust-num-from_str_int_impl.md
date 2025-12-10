---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

实现从&str解析成数字
- `"1234".parse::<u16>()` → 1234
- `"0010".parse::<u16>()` → 10
只要这个数字实现了FromStr trait
### Ⅱ. 实现层
parse
```rust
core::str

pub fn parse<F>(&self) -> Result<F, F::Err>  
where  
F: FromStr,
```



```rust
pub trait FromStr: Sized {
    type Err;
    fn from_str(s: &str) -> Result<Self, Self::Err>;
}
```

```rust
   pub const fn from_str_radix(src: &str, radix: u32) -> Result<$int_ty, ParseIntError> {
                <$int_ty>::from_ascii_radix(src.as_bytes(), radix)
   }
```
### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- 要点1  
- 要点2  

## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-str-trait-from_str](../str/traits/Rust-str-trait-from_str.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
