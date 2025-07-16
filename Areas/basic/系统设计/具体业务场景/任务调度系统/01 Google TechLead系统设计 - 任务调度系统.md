- 来源：https://www.bilibili.com/video/BV1A9w9eWEEq/?spm_id_from=333.1387.upload.video_card.click&vd_source=12808b3c6a27d423857284969f17ae7c

### clarify question
1.internal system
2.mvp
3.jobs can be all type

Funtioncal requirements
1.user can subumit tasks (task can be short-term temp script or long run service)
2.user can view task result on dashboard
3.advance features:
+shcedule some task in future(cron service)
- 未来任务/定时任务
+DAG (directed acyclic graph)job
- 有向无环图

Non Functional requirements
1.latency
- submit task exeuction within 10s
- dashboard sync within 602
2. scalabilty
3. reliability


job data model
1. repo(code/config/excutable binary)
2. metadata
- job id
- owner id 
- executable binary url
- input path
- output path
- created_time
- status
- num_of_retry

readt->waiting->running
- ->successs
- ->failde
	- -> final failure（>3次）
	- retry(<3次)
![](asserts/Pasted%20image%2020250714135226.png)
 
```
10 k user * 100*job/user/day*10 view
= 10k * 1000 request/100 k sec
=100 QPS
peak = 500 QPS

submission QPS = 10k*100/100k =10QPS
peak=50QPS
```
- 主从
- 读写分离
- 双机热备

三个信号判断需不需要Nosql和Sql
- 需不需要外键的支持
- 链表join查询
- 一致性
最终一致性来换取一个强一致性

关于Queue
- 持久性queue
- queue
	- 无状态，可以快速扩容

关于双写和分布式锁，data store和queue都考虑持久化需要考虑一致性问题，来引入前面提到的内容
- 额外的复杂性和风险
	- 分布式锁，写延迟增加，导致联级问题queue,挂了，data store不能写数据，实际是想解耦但是和初衷背道而驰。indisk收益部明显

另一种思路，从queue推导出data store状态。
- message queue as database
	- 技术上可以，不行还可以分区
	- 类似streaming system，延时低，sla保证
	- 坏处
		- 增加了工程师的心智负担
		- operation cost 是不是很高
- database as message queue
	- 优点
		- 在sapnner 实现了 pull 和poll语义
	- 缺点
		- 性能可能不是很高

work如何执行
![](asserts/Pasted%20image%2020250714142220.png)
方案一：发起rpc，pull一个任务，执行完成写回，、
- 好处：informer不用管理太多，fire-and-forget，任务丢出就不管
- 缺点：
	- 空转：worker不停的轮询，
	- worker挂掉：rpc是woekr发起的，worker有非常大的权限、
	- 并不是非常安全，work挂掉，无法更新
		- rpc通报上去，rpc多了会拥塞。加入timeout 多长时间没有执行完，就算job挂掉了
			- timeut 线上常驻
方案二：从informer这里发起rpc
- informer 算一个push mode，然后跟踪状态
- informer 定期查询状态，挂起，重新拉起
- 缺点：informer长期追踪worker的状态，额外建立长连接
- 额外知道worker地址

方案三：混合模型

- worker运行时，给它一个sidecar作为一个边车模式
- 融合了前两种的优势
- informer不再需要维护task和worker的映射，sidecar可以管理这些metadata
- 利用heartbeat来上报状态,三次包没有收到，按断是否进入retry和最终失败状态
- 缺点：增加了woeker的开销

资源
- add resource 增加资源
- exponential back_off retry
	- 2 4
	- 资源紧张是暂时的
- 系统层面：从runtime节省
	- 从runtime节省一些资源下来，变成一个容器
	- 容器会暴露更多的攻击面，添加隔离，使用gVisor
	- 裁剪的vm，googole的non-vm
		- 体积小于5m，保证安全性的同时，启动时间也很快
- 调度上的优化
	- 方案一：隔离：
		- io密集型（打满带宽）和cpu密集型（调用高性能机器）
		- 缺点
			- 浪费资源
	- 方案二：混合部署
		- 提高机器的利用效率
		- 又要专门开发调度算法
- recycle resource
	- AutoPilot的项目：回收资源
	- GoCrane的项目：自动回收一些资源
	- 可延后的资源：拷贝数据库之类的可以延后
	
### 复杂功能
- clitent封装一个库，cron service，维护一个优先级队列，向这个cron service，比较任务的执行时间，时间一到就弹出去，同时将任务的下一次执行时间插入到库里，进入下一次排队。
- DAG service
	- 排序后将任务丢给task client，等待返回信号，再提交下一个信号，之后再去提交下一个任务
- trade off 
	- 将cron service 和DAG service 放到informer
		- 不需要维护service，会降低复杂性
	- 额外的service会增加额外的开销，好处是可拓展性高，使用不同的方案实现，增加微服务就可以了
