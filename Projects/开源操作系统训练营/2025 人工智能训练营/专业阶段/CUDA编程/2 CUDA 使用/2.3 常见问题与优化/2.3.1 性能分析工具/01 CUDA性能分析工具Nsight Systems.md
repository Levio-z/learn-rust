### Nsight Systems（CLI）
>英伟达的系统级性能分析工具

启动profile
```
nsys profile -t cuda,nvtx,osrt -o target/add_cuda -f true target/add_cuda01
```
解析并统计性能信息
```
nsys stats target/add_cuda.nsys-rep
```
### 案例：
[02 CUDA入门案例优化01-核函数](../../2.2%20使用示例/2.2.1%20入门案例-加法函数/02%20CUDA入门案例优化01-核函数.md)