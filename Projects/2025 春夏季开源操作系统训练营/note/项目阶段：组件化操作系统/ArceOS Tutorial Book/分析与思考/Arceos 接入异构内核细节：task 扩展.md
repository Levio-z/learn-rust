
### 背景
接入异构内核，task数据结构需要添加不同内容来适配
### 方案一，通过feature开启字段生成
实现：
- 使用feature控制字段是否开启
	示例:
- ![Pasted image 20250521175842](Pasted%20image%2020250521175842.png)
	分析：
	- 不会有性能影响
	- 不利于可读性和异构扩展性
		- 可读性：不同组件数据相互耦合
		- 扩展性：作为底层结构被其他组件依赖，一个特性导致组件字段的修改会影响其他组件
### 方案二：**利用索引机制**
实现：
- 将**扩展内容额外实现在新的结构中**
- 两者通过某一个共通**字段关联**
	- 通过id在map中查询扩展结构体![Pasted image 20250521180429.png](Pasted%20image%2020250521180429.png)]
	分析：
- 查询索引的过程会带来性能开销

### 方案三：简化版 TLS 机制 —— 引入 extension 扩展机制
- 实现：
	- 为 Task 引入一个 extension 域
	- 当外部实现了扩展内容，可初始化 extension 域
- 示例：
 ![Pasted image 20250521180728.png](Pasted%20image%2020250521180728.png)
  分析：
- **使用指针进行调用，和传统结构体的访存开销近似**
- 在保证扩展性的同时不影响性能开销
