repo:https://github.com/InfiniTensor/llaisys
![](asserts/Pasted%20image%2020250819094341.png)

![](asserts/Pasted%20image%2020250819094602.png)


![](asserts/Pasted%20image%2020250819095023.png)
- message 是list
- 可以设置流式

### 单人推理服务
![](asserts/Pasted%20image%2020250819095416.png)

### 会话管理
![](asserts/Pasted%20image%2020250819095906.png)

### 单人推理服务
![](asserts/Pasted%20image%2020250819100817.png)
![](asserts/Pasted%20image%2020250819100911.png)


### 多人推理服务
![](asserts/Pasted%20image%2020250819101001.png)
服务多请求
![](asserts/Pasted%20image%2020250819101351.png)

![](asserts/Pasted%20image%2020250819101413.png)

![](asserts/Pasted%20image%2020250819101635.png)
**希望在迭代训练时，不是一次性把数据集分成 batch，而是每次迭代按需动态生成一个 batch**，也就是 “**迭代为单位形成 batch**”

![](asserts/Pasted%20image%2020250819102124.png)

### 连续批次
![](asserts/Pasted%20image%2020250819102229.png)

https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com

![](asserts/Pasted%20image%2020250819102420.png)


- 请求池可能是新来的也有推了好几轮的，形成批次，进行一轮推理，再返回请求池

每轮请求
- 请求信息
- KVCacnhe绑定
- IO管线
### KVCache的内存占用很大
![](asserts/Pasted%20image%2020250819103057.png)
- 参数、层数、上下文长度
### Paged Attention （vllm的核心创新）
![](asserts/Pasted%20image%2020250819103333.png)
#### Attention好处之一-内存共享
![](asserts/Pasted%20image%2020250819103638.png)

![](asserts/Pasted%20image%2020250819103803.png)
- 2. 没有用满，复制，直接写会产生冲突

### Beam Search
![](asserts/Pasted%20image%2020250819103925.png)


