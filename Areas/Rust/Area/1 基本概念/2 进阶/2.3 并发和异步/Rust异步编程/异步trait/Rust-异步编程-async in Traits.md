---
tags:
  - permanent
---
## 1. 核心观点
- 夜间版本：目前，Rust 稳定版中无法使用`异步 fn` 在特征中。自 2022 年 11 月 17 日起，async-fn-in-trait 的 MVP 可在编译器工具链的夜间版本中提供， [详情请见此处](https://blog.rust-lang.org/inside-rust/2022/11/17/async-fn-in-trait-nightly.html) 。

- 稳定版替代方案：与此同时，有一个关于稳定工具链的替代方案，使用 异[步特征笼子 crates.io](https://github.com/dtolnay/async-trait)。

>注意，使用**这些特征方法会导致每个函数调用进行堆分配**。这对绝大多数应用来说成本不高，但在决定是否在预期每秒被调用数百万次的低层函数的公共 API 中使用此功能时，应考虑这一点。
## 2. 背景 / 出处




## 4. 与其他卡片的关联


## 5. 应用 / 启发




## 6. 待办 / 进一步探索
