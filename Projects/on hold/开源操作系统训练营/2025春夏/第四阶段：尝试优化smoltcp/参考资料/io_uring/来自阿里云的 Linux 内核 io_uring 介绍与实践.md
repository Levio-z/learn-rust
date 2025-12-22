### io_uring 背景
AIO？
异步IO提供了高性能实现的基础
- AIO现状
	- 不够异步
	- 仅支持Direct IO
		- buffer IO
		- Direct IO 直接访问磁盘，文件系统
	- IO请求元数据开销较大
		- 本身设计不够精简
	- IOPOLL支持不好
		- 完成通知方式
			- 中断
			- 轮询，后一种就是IOPOLL
				- 利用CPU资源换取一点延迟，io比较小
- 使用方便： 三个系统调用，liburing用户态库编程良好
- ![](../../asserts/Pasted%20image%2020250529202015.png)
- 特性丰富：
- 高性能：IO请求overhead
### 性能对比
![](../../asserts/Pasted%20image%2020250529202445.png)
### io_uring整体架构
![](../../asserts/Pasted%20image%2020250529202713.png)
- 不需要系统调用
	- aio只有完成队列，uring完成解耦
![](../../asserts/Pasted%20image%2020250529202927.png)
SQPOLL
- 内核态创建线程提交

