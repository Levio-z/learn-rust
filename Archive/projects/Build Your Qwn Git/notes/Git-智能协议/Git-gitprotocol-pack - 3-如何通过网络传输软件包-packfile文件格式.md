---
tags:
  - permanent
---
## 1. 核心观点  

#### 定义




## 2. 背景/出处  
- 来源：
	- https://git-scm.com/docs/gitformat-pack
	- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/git-upload
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 格式
### （`0008NAK` ）
- 是 Git 协议协商阶段的一部分。它表示服务器没有找到客户端和服务器之间针对所请求引用的任何共同提交。这种情况通常发生在初始克隆期间，或者客户端还没有任何对象时。
#### header
```
4-byte signature:
    The signature is: {'P', 'A', 'C', 'K'}

   4-byte version number (network byte order):
Git currently accepts version number 2 or 3 but
       generates version 2 only.

4-byte number of objects contained in the pack (network byte order)

Observation: we cannot have more than 4G versions ;-) and
more than 4G objects in a pack.
```
#### body
```
(undeltified representation)
n-byte type and length (3-bit type, (n-1)*7+4-bit length)
compressed data

(deltified representation)
   n-byte type and length (3-bit type, (n-1)*7+4-bit length)
   base object name if OBJ_REF_DELTA or a negative relative
offset from the delta object's position in the pack if this
is an OBJ_OFS_DELTA object
   compressed delta data

Observation: the length of each object is encoded in a variable
length format and is not constrained to 32-bit or anything.
```
n-byte type and length (3-bit type, (n-1)*7+4-bit length)
[Git-Parsing packfiles  object entry 格式](../notes/deltification%20概念/Git-Parsing%20packfiles%20%20object%20entry%20格式.md)
[Git-deltification 概念-TOC](../deltification%20概念/Git-deltification%20概念-TOC.md)
[RefDelta 如何确定基础对象类型](../Reference/RefDelta%20如何确定基础对象类型.md)
#### 校验和
校验和

### 实际
https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/git-upload

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：
	- 【ref】[智能协议测试](智能协议测试.md)
	- [Git-URL Format  URL 格式](Git-URL%20Format%20%20URL%20格式.md)
	- [Git-gitprotocol-pack - pkt-line 格式](Git-gitprotocol-pack%20-%20pkt-line%20格式.md)
## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
