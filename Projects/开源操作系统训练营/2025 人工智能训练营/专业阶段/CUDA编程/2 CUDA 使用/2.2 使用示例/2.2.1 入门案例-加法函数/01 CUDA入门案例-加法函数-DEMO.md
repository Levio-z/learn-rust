### cpu加法函数
```
#include <stdio.h>

#include <iostream>

#include <vector>

  

const size_t SIZE = 1 <<20;

void add_cpu(std::vector<float> &c,const std::vector<float> &a,const std::vector<float> &b){

    for (size_t i =0;i< a.size();i++){

        c[i] = a[i] +b[i];

    }

  

    std::cout << "执行完毕" << '\n';

    std::cout <<"c[SIZE-1]:"<< c[SIZE-1] << '\n';

}

  

int main(){

    std::vector<float> a(SIZE,1);

    std::vector<float> b(SIZE,2);

    std::vector<float> c(SIZE,0);

  

    add_cpu(c,a,b);

    return 0;

}
```
1. 准备和初始化数据 
2. 定义加法函数 • 靠循环来进行所有的元素加法 
3. 调用函数 
4. 验证结果

```
g++ add_cpu.cpp -o tatget/add_cpu
```

### cpu->gpu
#### 步骤
1. 准备和初始化数据（CPU) 
2. 数据传输到 GPU 
3. GPU 从 GM 中读取并计算后写回 （调用函数计算） 
4. 将 GPU 数据传输回 CPU 
5. 验证结果

##### 核心：怎么编写加法函数
目标：要写在 GPU 上运行的加法函数并 进行调用解决： 

解决
- 定义 block 的数量和大小来指挥线程同时进行/并行计算
- -定义 GPU 上的加法函数 
- 结合定义的信息调用 GPU 加法函数
```c++
    // Set execution configuration parameters

    //      grid_dim: number of CUDA threads per grid block

    //      block_dim: number of blocks in grid

    dim3 block_dim(256);

    dim3 grid_dim((SIZE + block_dim - 1) / block_dim);

  
  

    // 3.GPU reads from global memory, performs computation, and writes back (invoke computation function)

    // call the cuda add kernel

    add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);
```
- dim3: CUDA 表示线程层级结构的类型 
- <<<>>>: 传递线程层级信息给核函数 
```c++
/**
 * @brief 并行计算两个数组元素逐个相加的结果，存储到目标数组中。
 * 
 * @tparam T 参与运算的数据类型，要求支持加法操作（operator+）。
 * @param[out] c 目标数组指针，存储结果。数组大小至少为 n。
 * @param[in] a 第一个加数数组指针，数组大小至少为 n。
 * @param[in] b 第二个加数数组指针，数组大小至少为 n。
 * @param[in] n 要处理的元素数量。
 */
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

```
- 核函数（kernel）：设备侧的入口函数 
- __global__: 表示这是个核函数 
- blockIdx: 表示刚刚提到的 block 的编号 
- blockDim: 表示刚刚提到的 block 的大小
- threadIdx: 表示刚刚提到的 thread 的编号

>cuda为什么不支持vector，因为它分配在host端，设备端内存不能直接用主机端结构


怎么编译运行？用 gcc 报错了… 因为要用：NVCC！也就是 CUDA 的编 译器（NIVIDIA CUDA Compiler） 
• CUDA Toolkit 的一部分

##### 验证

编译：
```
nvcc add_gpu/add_gpu.cu -o target/add_gpu
```
执行：
```
./target
```
>注意备注.gitignore忽略target文件夹
- 真的跑起来了？
- 看GPU利用率
- 运行程序同时使用nvidia-smi
```
watch -n 1 nvidia-smi
```
- 下面是进程
- 中间显示显存使用？
[1 watch -n 1 nvidia-smi](chatgpt/1%20watch%20-n%201%20nvidia-smi.md)

