---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层
- [① From for Shapes（向上注入）](#①%20From%20for%20Shapes（向上注入）)
- [② TryInto for Shapes（向下投影）](#②%20TryInto%20for%20Shapes（向下投影）)
- [③ impl Shape for Shapes（行为分派）](#③%20impl%20Shape%20for%20Shapes（行为分派）)

核心：
```rust
impl MyTrait for RespEnum {
    #[inline]
    async fn do_something(&self) {
        match self {
            RespEnum::RespNull(inner) => MyTrait::do_something(inner).await,
            RespEnum::RespSet(inner) => MyTrait::do_something(inner).await,
        }
    }
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
- https://github.com/learn-rust-projects/rust-lab/blob/master/enum/enum_dispatch_lab/fixtures/exmaple01_expand.rs



#### ① `From<T> for Shapes`（向上注入）

```rust
impl From<Circle> for Shapes { ... }
impl From<Rectangle> for Shapes { ... }
impl From<Triangle> for Shapes { ... }
```

**作用**
- 允许 `Circle / Rectangle / Triangle` **无损注入** 到 `Shapes`
- 支持 `.into()` / `Shapes::from(x)`
- 这是 **enum 作为 sum type 的标准“构造方向”**

**本质**

> enum 的每个 variant = 一个构造器  
> `From<T>` = 构造器的 trait 化表达

---

#### ② `TryInto<T> for Shapes`（向下投影）

```rust
impl TryInto<Circle> for Shapes { ... }
impl TryInto<Rectangle> for Shapes { ... }
impl TryInto<Triangle> for Shapes { ... }
```

**作用**

- 尝试从 `Shapes` 中“取回”具体 variant
- 失败即报错（`Err(&'static str)`）

**设计含义（非常重要）**

- enum → variant **不是总是安全的**
    
- 必须是 `TryInto` 而不是 `Into`
    
- 这就是 **代数数据类型（ADT）中 sum → component 的经典语义**
    

---

#### ③ `impl Shape for Shapes`（行为分派）

```rust
impl Shape for Shapes {
    fn area(&self) -> f64 { match self { ... } }
    fn perimeter(&self) -> f64 { match self { ... } }
}
```

**作用**

- 让 `Shapes` 成为一个 **trait object 的静态替代品**
    
- 每个方法 = 一次 `match` + 静态分发
    

**关键点**

```rust
Shapes::Circle(inner) => Shape::area(inner)
```

不是虚表，不是动态分派，而是：

> **enum + match + trait 的零成本多态**



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
