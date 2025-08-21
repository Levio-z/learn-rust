### 内存层级结构
![](asserts/Pasted%20image%2020250820095449.png)
◆ 解决方案：阶层结构 
◆ 越靠近上层/处理器/计算单元 →越快、越小、越贵 → 存放更常用的数据和指令 
◆ 越靠近下层 →越慢、 越大、越便宜 → 存放更不常用/更大规模的数据和指令
### GPU内存单元-A100
![](asserts/Pasted%20image%2020250813150954.png)
中间两个蓝色的是L2级缓存
### 浅看GPU硬件单元
![](asserts/Pasted%20image%2020250813151050.png)
### GPU内存结构
![](asserts/Pasted%20image%2020250813151212.png)
https://ai.gopubby.com/memory-types-in-gpu-6373b7a0ca47?gi=06646da76c95

- 寄存器：离处理器核心最近的单元
- 共享内存：192kb，每一个SM共用的
- L1缓存：
	- SM最上面的L1 instruction cache
	- 下面也存在
- 只读缓存：
	- TEX：常量缓存和纹理缓存
- L2缓存
	- 所有SM共用的
- 全局内存
- CPU内存
### CUDA内存模型（逻辑）
![](asserts/Pasted%20image%2020250813151737.png)
https://ai.gopubby.com/memory-types-in-gpu-6373b7a0ca47?gi=06646da76c95
- 本地内存：逻辑视图上在block'内
	- **重要**：名字叫 “local memory”，但 **它不在寄存器内，也不在共享内存内，而是在全局内存（device memory）中分配**。
		- 寄存器存在本地内存，从on-chip到off-chip到gpu最慢的内存上
- 全局内存：和上一页设备内存不是完美对应，不是设备侧内存
	- 常量内存和纹理内存在全局内存里面
	- [9 常量内存和纹理内存](../../../chatgpt/9%20常量内存和纹理内存.md)
		- 专门为他们设计了独立的缓存，每个SM都有

