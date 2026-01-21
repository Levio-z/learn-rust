---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

**功能：用于检查Rust依赖安全性的Cargo插件
- 应用场景：
    - 生产环境/商业项目中检查依赖授权协议
    - 扫描供应链中的已知安全问题
- 典型问题：
    - 检测GPL v3等"传染性"协议依赖（使用此类依赖会导致商业代码必须开源）
    - 识别存在安全漏洞的依赖版本
- 安装命令：
```
﻿cargo install --locked cargo-deny
﻿cargo install cargo-deny --version 0.13.1
```

### Ⅱ. 应用层
- https://github.com/learn-rust-projects/template/blob/master/deny.toml
    - 常见许可类型：
        - MIT/Apache-2.0：商业友好许可
        - BSD-2/3-Clause：宽松开源许可
        - MPL-2.0：弱copyleft许可
        - CC0-1.0：公共领域许可
    - 配置方式：在deny.toml中**通过allow列表指定允许的许可证**
    - 添加新许可：当遇到未授权的许可证时，可将其SPDX标识符添加到allow列表
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
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
