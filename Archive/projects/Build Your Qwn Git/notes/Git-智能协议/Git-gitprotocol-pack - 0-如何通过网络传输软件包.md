---
tags:
  - permanent
---
## 1. 核心观点  

Git 支持通过 ssh://、git://、http:// 和 file:// 传输方式在 pack 文件中传输数据。存在两组协议，一组用于将数据从客户端推送到服务器，另一组用于将数据从服务器拉取到客户端。这三种传输方式（ssh、git 和 file）使用相同的协议来传输数据。http 协议的文档请参见 [gitprotocol-http[5]](https://git-scm.com/docs/gitprotocol-http) 。

规范的 Git 实现中调用的进程是 _upload-pack。_ 
- 服务器端使用 _fetch-pack_ 获取数据，客户端使用 fetch-pack 获取数据；
- 服务器端使用 _receive-pack_ 推送数据，客户端使用 _send-pack_ 推送数据。

- 该协议的功能是让服务器告知客户端服务器上的当前数据，然后双方协商发送最少的数据量，以完成对其中一方的完整更新。

## 2. 背景/出处  
- 来源：https://git-scm.com/docs/gitprotocol-pack
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

当一个 Git 仓库需要获取另一个仓库拥有的数据时，前者可以从后者_获取数据_ 。**此操作会确定服务器拥有而客户端没有的数据，然后将这些数据以 packfile 格式流式传输给客户端。**

### Reference Discovery
#### 前置内容
[Git-智能协议基本概念-TOC](Git-智能协议基本概念-TOC.md)
[Git-gitprotocol-pack - pkt-line 格式](Git-gitprotocol-pack%20-%20pkt-line%20格式.md)
#### [Git-gitprotocol-pack - 0-如何通过网络传输软件包](Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)
- 1、客户端获取ref列表：[Git-gitprotocol-pack - 1-如何通过网络传输软件包-获取服务器端的引用（refs）列表及 capabilities 信息](Git-gitprotocol-pack%20-%201-如何通过网络传输软件包-获取服务器端的引用（refs）列表及%20capabilities%20信息.md)
- 2、客户通过want请求获取packfile：[Git-gitprotocol-pack - 2-如何通过网络传输软件包-协商阶段请求packfile文件](Git-gitprotocol-pack%20-%202-如何通过网络传输软件包-协商阶段请求packfile文件.md)
- 3、解析packfie

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- 【ref】[智能协议测试](智能协议测试.md)
	- [Git-URL Format  URL 格式](Git-URL%20Format%20%20URL%20格式.md)
	- [Git-gitprotocol-pack - pkt-line 格式](Git-gitprotocol-pack%20-%20pkt-line%20格式.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
