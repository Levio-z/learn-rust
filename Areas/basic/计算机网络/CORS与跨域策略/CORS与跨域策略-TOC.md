# CORS 与跨域策略知识目录 (TOC)

## 1️⃣ 基础概念
- 1.1 同源策略 (Same-Origin Policy, SOP)
- 1.2 [跨域 (Cross-Origin) 概念](../../../../book/图解Http/notes/Archieve/跨域-基本概念.md)与问题场景
	- 1.2.1 [跨域是谁来识别和执行的](../../../../book/图解Http/notes/Archieve/跨域-跨域是谁来识别和执行的.md)

## 2️⃣ CORS 基础
- 2.1 CORS 原理与浏览器安全机制
- 2.2 主要 HTTP Header
  - Access-Control-Allow-Origin
  - Access-Control-Allow-Methods
  - Access-Control-Allow-Headers
  - Access-Control-Allow-Credentials
  - Access-Control-Expose-Headers
  - Access-Control-Max-Age
- 2.3 请求类型
  - 简单请求 (Simple Request)
  - 预检请求 (Preflight Request)

## 3️⃣ 跨域策略实现
- 3.1 服务器端配置
  - Web 框架中间件
  - Nginx 配置
- 3.2 前端解决方案
  - JSONP
  - 代理服务器
  - postMessage / iframe 通信

## 4️⃣ Rust 实战框架
- 4.1 Actix Web CORS 配置
- 4.2 Axum CORS 配置
- 4.3 warp / 其他框架的跨域处理

## 5️⃣ 高级实践与注意事项
- 5.1 带 Cookie 的跨域请求配置
- 5.2 安全策略与风险防护
  - 避免使用 *
  - 精确域名与方法配置
  - 防止敏感信息泄露

## 6️⃣ 面试/项目加分点
- 6.1 理解 SOP 与 CORS 原理
- 6.2 区分简单请求与预检请求
- 6.3 Rust 框架中实现安全跨域策略
- 6.4 带 Cookie 跨域请求注意事项
