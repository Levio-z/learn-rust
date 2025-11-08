---
tags:
  - fleeting
---
## 1. 核心观点  
用一句话写出本卡片的核心思想，越简洁越好。

## 2. 背景/出处  
- 来源：
	- https://github.com/Levio-z/learning-cxx/blob/main/drafts/01_epoll_edge_blocking/README.md
	- https://xiaolincoding.com/os/8_network_system/selete_poll_epoll.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- 要点1  
- 要点2  

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 面试题回答出来 
- [ ] io多路复用在Rust中或者其他技术的应用
- [ ] 好奇的问题，但是可能不重要，只是记录
	- [ ] os套接字是否需要解决持续通信
	- [ ] socket这些函数为什么这么设计
	- [ ] os底层维护的数据结构是什么样的
	- [ ] 进程间共享的部分和不共享的部分
	- [ ] **边缘触发模式一般和非阻塞 I/O 搭配使用**，程序会一直执行 I/O 操作，直到系统调用（如 `read` 和 `write`）返回错误，错误类型为 `EAGAIN` 或 `EWOULDBLOCK`。
	- [ ] 一般来说，边缘触发的效率比水平触发的效率要高，因为边缘触发可以减少 epoll_wait 的系统调用次数，系统调用也是有一定的开销的的，毕竟也存在上下文的切换
- [ ] 资源，不一定看
	- [ ] epoll lt / et 模式区别：https://wenfh2020.com/2020/06/11/epoll-lt-et/
	- [ ] 源码：https://wenfh2020.com/2020/04/23/epoll-code/
