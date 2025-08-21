```c++
// 核函数声明（模板函数定义一般放头文件，这里为演示直接写在一起）

template <typename T>

__global__ void sum_kernel(T *result, const T *input, size_t n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {

        // 注意：这里有“竞争条件”问题！多个线程同时操作 *result，

        // 实际场景应改用原子操作（如 atomicAdd）或归约（reduction）优化

        *result += input[idx];

    }

}
```

### 测试
数据大小：const size_t SIZE = 1 << 20; // 元素总数 half个数
```
nvcc sum_gpu/sum_gpu.cu -o target/sum_gpu
```

```
nsys profile -t cuda,nvtx,osrt -o target/sum_gpu -f true target/sum_gpu
```

```
nsys stats target/sum_gpu.nsys-rep
```

```
./target/sum_gpu
```

结果：
```
GPU result: 28.0194
CPU result: 524282
```
### 问题
疑问：为什么错了？ 
答案：
◆ 因为这是并行！ 
◆ 多个线程同时进行写入操作！ 
◆ 但之前的加法也是并行啊？为什么能对？

![](asserts/Pasted%20image%2020250813111505.png)


![](asserts/Pasted%20image%2020250813111514.png)

答案：
◆ 因为之前的加法是尴尬并行（Embarrassingly Parallel） 
→ 之前的加法没有冲突/竞争的数据依赖问题

疑问：那要怎么解决这个竞争的问题？

答案： 
◆ 确保顺序执行，同时只能有一个写入操作 
**→ 原子操作**
→ “either run to completion in one go, or have no effect”