---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
	- https://docs.rs/tower/latest/tower/trait.Service.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
pub trait Service<Request> {
    type Response;
    type Error;
    type Future: Future<Output = Result<Self::Response, Self::Error>>;

    // Required methods
    fn poll_ready(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), Self::Error>>;
    fn call(&mut self, req: Request) -> Self::Future;
}
```
- 基本定义: 表示从Request到Response的异步函数，建模网络应用的核心接口
    - 关联类型:
        - type Response: 成功响应类型
        - type Error: 错误类型
        - type Future: 表示异步处理结果的Future
    - 关键方法:
        - poll_ready: 检查服务是否准备好处理请求
        - call: 处理请求并返回Future
    - Future输出: 必须为Result<Self::Response, Self::Error>

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
