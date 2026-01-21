---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
 - 典型流程：
	- 生成SigningKey（包含密钥对）
	- 调用sign()方法生成签名
	- 通过verify()方法验证签名
- 接口特点：
	- 支持统一密钥操作（自动派生验证密钥）
	- 提供多种序列化格式（PKCS#8/PEM等）
- 安全注意：
	- 上下文字符串应全局唯一
	- 密钥材料需足够随机（非简单密码）
- 签名生成:
    - 使用SigningKey::generate()生成包含公私钥对的签名密钥
    - 通过signing_key.sign(message)对消息进行签名
- 签名验证:
    - 使用signing_key.verify(message, signature)验证签名有效性
    - 通过signing_key.verifying_key()获取验证公钥
- 序列化:
    - 使用to_bytes()方法将密钥和签名序列化为字节数组
    - 注意：私钥(SECRET_KEY_LENGTH)不应传输，只需公开公钥(PUBLIC_KEY_LENGTH)

### Ⅱ. 应用层

```rust
cargo add ed25519_dalek --features rand_core

```


### Ⅲ. 实现层

### **IV**.原理层
- 密钥体系：
	- 包含secret key（签名用）和public key（验证用）
	- public key可从secret key派生获得
- 算法对比：
	- 同属椭圆曲线加密家族（与ECDSA/RSA并列）
	- 相比RSA具有更短的密钥长度（256bit）
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
 
  
