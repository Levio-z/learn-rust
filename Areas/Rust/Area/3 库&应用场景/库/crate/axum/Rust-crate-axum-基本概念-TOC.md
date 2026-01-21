---
tags:
  - notre
---
## 1. 核心观点  
### Ⅰ. 概念层

- 生态兼容性: 完全兼容tower生态系统，tower是Rust中强大的服务抽象层，**定义了Service trait标准接口**
- 中间件优势: 直接复用tower-http生态的中间件（超时、追踪、压缩、鉴权等），无需自行开发
- 互操作性: 可与hyper或tonic等基于tower的框架共享中间件，实现无缝集成
- 设计理念: 采用"服务即trait"的设计模式，所有组件都实现Service trait保证一致性

### axum的互操作性与模块化设计
- GRPC集成: 与tonic（Rust的GRPC实现）互操作方便，共享hyper和tower底层
- 混合部署: 支持与AWS smithy等HTTP服务共存于同一应用，路由可灵活分配
- 模块化架构: 通过tower::Service抽象实现高度解耦，各组件可独立替换
- 生态定位: 作为hyper的上层封装，保持轻量级的同时提供完整Web框架功能
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 添加依赖
```
cargo add axum -F http2,query,tracing
```
### axum的性能与安全性
- 性能表现: 作为hyper的薄封装层，性能与hyper基本一致（平均延迟0.65ms）
- 稳定性: 延迟标准差仅0.36ms，远低于actix-web的0.92ms，表现更稳定
- 安全保证: 使用#![forbid(unsafe_code)]强制全安全Rust实现，无内存安全风险


- 性能对比：axum与其他框架
	- **最稳定**。这两者性能几乎一致（axum 基于 hyper）。它们的**标准差（Stdev）最低**，最大延迟也最低（约 13-16ms），说明性能曲线非常平滑，极少出现卡顿。
	- 第一梯队: actix-web(933k)、axum(768k)、hyper(768k)属于性能第一阵营
	- 延迟分布: axum最大延迟仅16.41ms，远优于actix-web的152.70ms
	- 综合测试: 在复杂场景下达到143k请求/秒，与actix-web的184k差距不大
	- 选择考量: **牺牲少量性能换取更好的安全性（无unsafe代码）和开发体验**

- https://web-frameworks-benchmark.netlify.app/result?asc=0&l=rust&order_by=level512
### 处理handler的参数
- 看https://docs.rs/axum/latest/axum/extract/index.html
### 调试
- 所有handler的错误都和函数签名有关系
-  忘记添加async关键字导致Handler trait不满足
	- 错误信息："the trait bound `fn() -> &'static str {index_hanlder}: Handler<,> is not satisfied"
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-crate-axum-示例代码](note/atomic/Rust-crate-axum-示例代码.md)
	- [Rust-crate-axum-http-路由绑定](note/atomic/Rust-crate-axum-http-路由绑定.md)
- 相似主题：
	- 

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
