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
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
代码地址：https://github.com/tokio-rs/axum
- 路由定义:
	- 使用Router::new().route()链式调用定义路由
	- 支持HTTP方法限定（get/post等），每个路由对应async handler
 - Handler规范:
	- 必须是async函数，否则会编译错误
	- 参数通过提取器（如`Json<T>`）自动解析请求数据
	- **返回值需实现IntoResponse trait（字符串/元组等已内置实现）在frame层做转换**
- 请求处理:
	- 基础路由直接返回字符串（自动设置200状态码）
	- 复杂路由可返回(StatusCode, Json)元组定制响应
	- 自动处理Content-Type等头部信息（如application/json）
 - 类型安全:
	- 使用`#[derive(Serialize, Deserialize)]`自动生成序列化代码
	- 编译时检查请求/响应类型，避免运行时错误
- 只要任务参数实现了fromrequestpart


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
