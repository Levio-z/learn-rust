
### 设备信息
疑问： 
◆ 启用越多的线程 → 越大的并行度 → 越好的性能 
◆ 那最多能同时启用多少线程呢？→ 硬件相关，需要获取设备信息

- 网络/官网查询，硬件文档

- 代码中获取
	- cudaDeviceProp:

- maxGridSize
	- int[3] x, y, z 三个方向分别最 多可支持的 block 数
- maxThreadsDim
	- int[3]
	- 每个 Block 中 x, y, z 三 个方向分别最多可支持 的线程数
- maxThreadsPerBlock
	- 每个 block 中最多可有 的线程数
	![](asserts/Pasted%20image%2020250812145355.png)
### cuda版本
```
nvcc -version
```

nvidia是驱动支持的最高版本，不一定是正在使用的版本

