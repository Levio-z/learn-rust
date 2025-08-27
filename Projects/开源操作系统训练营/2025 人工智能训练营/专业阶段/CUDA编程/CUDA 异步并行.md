![](asserts/Pasted%20image%2020250822082349.png)
![](asserts/Pasted%20image%2020250822082428.png)

![](asserts/Pasted%20image%2020250822082618.png)

![](asserts/Pasted%20image%2020250822082724.png)

![](asserts/Pasted%20image%2020250822082739.png)
![](asserts/Pasted%20image%2020250822082851.png)



![](asserts/Pasted%20image%2020250822082953.png)
tensor 加速混合精度计算

![](asserts/Pasted%20image%2020250822083154.png)




![](asserts/Pasted%20image%2020250822083300.png)



![](asserts/Pasted%20image%2020250822083431.png)



![](asserts/Pasted%20image%2020250822083613.png)

```c++
#include <cuda/api_wrappers.hpp>
#include <iostream>
#include <vector>

// 假设的计算函数 - Consumer角色
__global__ void compute_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f + 1.0f; // 示例计算
    }
}

// 初始化设备
cuda::device::id_t init_device(int device_id = 0) {
    cuda::device::id_t device(device_id);
    cuda::device::set(device);
    return device;
}

int main() {
    try {
        // 初始化
        auto device = init_device();
        const size_t batch_size = 1024 * 1024; // 每批次数据大小
        const size_t num_batches = 10;         // 总批次数
        const size_t threads_per_block = 256;
        const size_t blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

        // 主机内存分配
        std::vector<float> h_input(batch_size * num_batches);
        std::vector<float> h_output(batch_size * num_batches);
        
        // 初始化输入数据
        for (size_t i = 0; i < h_input.size(); ++i) {
            h_input[i] = static_cast<float>(i) / 1000.0f;
        }

        // 设备内存分配
        float* d_input;
        float* d_output;
        cuda::malloc(&d_input, batch_size * num_batches * sizeof(float));
        cuda::malloc(&d_output, batch_size * num_batches * sizeof(float));

        // 创建流水线
        cuda::pipeline::handle_t pipeline;
        cuda::pipeline::create(&pipeline);

        // 外层循环：处理所有批次
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t offset = batch * batch_size;
            
            // Acquire - 获取流水线资源（线程协作）
            cuda::pipeline::acquire(pipeline);

            // Producer角色：异步数据拷贝
            cuda::memcpy_async(
                d_input + offset, 
                h_input.data() + offset, 
                batch_size * sizeof(float),
                cuda::memcpy_kind::host_to_device,
                pipeline
            );

            // Submit - 提交拷贝任务（线程独立）
            cuda::pipeline::submit(pipeline);

            // Commit - 确认提交（线程协作）
            cuda::pipeline::commit(pipeline);

            // Wait - 等待前序操作完成（线程协作）
            cuda::pipeline::wait(pipeline);

            // Consumer角色：执行计算（线程独立）
            compute_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_input + offset, 
                d_output + offset, 
                batch_size
            );

            // 将结果拷贝回主机
            cuda::memcpy_async(
                h_output.data() + offset,
                d_output + offset,
                batch_size * sizeof(float),
                cuda::memcpy_kind::device_to_host,
                pipeline
            );

            // Release - 释放流水线资源（线程协作）
            cuda::pipeline::release(pipeline);
        }

        // 等待所有操作完成
        cuda::device::synchronize(device);

        // 验证结果（省略）
        std::cout << "流水线执行完成" << std::endl;

        // 资源释放
        cuda::free(d_input);
        cuda::free(d_output);
        cuda::pipeline::destroy(pipeline);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

```


![](asserts/Pasted%20image%2020250822084136.png)
### 同步加载
![](asserts/Pasted%20image%2020250822091831.png)

![](asserts/Pasted%20image%2020250822092001.png)


```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

// 带计时功能的同步数据拷贝内核
template <typename T>
__global__ void sync_copy_kernel(
    T *global, 
    T *output, 
    uint64_t *clock, 
    size_t copy_count, 
    size_t total_elements, 
    extern __shared__ char s[]
) {
    // 将共享内存char指针转换为模板类型T的指针
    T *shared = reinterpret_cast<T*>(s);
    
    // 计算当前线程块的全局内存偏移量
    uint64_t block_offset = blockIdx.x * blockDim.x * copy_count;
    
    // 记录开始时间
    uint64_t clock_start = clock64();
    
    // 全局内存 -> 共享内存拷贝
    for (size_t i = 0; i < copy_count; ++i) {
        const size_t local_idx = blockDim.x * i + threadIdx.x;
        const size_t global_idx = block_offset + local_idx;
        if (global_idx < total_elements) {
            shared[local_idx] = global[global_idx];
        }
    }
    
    // 等待所有线程完成共享内存写入
    __syncthreads();
    
    // 记录结束时间并累加计时结果（仅由线程块的第一个线程执行）
    uint64_t clock_end = clock64();
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(clock), clock_end - clock_start);
    }
    
    // 共享内存 -> 全局内存写回
    for (size_t i = 0; i < copy_count; ++i) {
        const size_t local_idx = blockDim.x * i + threadIdx.x;
        const size_t global_idx = block_offset + local_idx;
        if (global_idx < total_elements) {
            output[global_idx] = shared[local_idx];
        }
    }
}

// 检查CUDA操作是否成功
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    // 配置参数
    const size_t total_elements = 1024 * 1024 * 32;  // 总元素数量
    const size_t threads_per_block = 256;            // 每个线程块的线程数
    const size_t copy_count = 4;                     // 每个线程块的拷贝批次
    using DataType = float;                          // 数据类型（可改为int, double等）
    
    // 计算线程块数量
    const size_t elements_per_block = threads_per_block * copy_count;
    const size_t blocks_per_grid = (total_elements + elements_per_block - 1) / elements_per_block;
    
    // 计算共享内存大小
    const size_t shared_mem_size = elements_per_block * sizeof(DataType);
    
    // 主机内存分配与初始化
    std::vector<DataType> h_input(total_elements);
    std::vector<DataType> h_output(total_elements, 0);
    for (size_t i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<DataType>(i) / 1000.0f;  // 简单初始化
    }
    
    // 设备内存分配
    DataType *d_input, *d_output;
    uint64_t *d_clock;
    CHECK(cudaMalloc(&d_input, total_elements * sizeof(DataType)));
    CHECK(cudaMalloc(&d_output, total_elements * sizeof(DataType)));
    CHECK(cudaMalloc(&d_clock, sizeof(uint64_t)));
    
    // 初始化设备计时变量
    CHECK(cudaMemset(d_clock, 0, sizeof(uint64_t)));
    
    // 数据从主机拷贝到设备
    CHECK(cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(DataType), cudaMemcpyHostToDevice));
    
    // 启动内核
    sync_copy_kernel<DataType><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_input,
        d_output,
        d_clock,
        copy_count,
        total_elements,
        nullptr  // 共享内存由内核参数指定，这里传nullptr
    );
    CHECK(cudaGetLastError());  // 检查内核启动错误
    
    // 等待内核执行完成
    CHECK(cudaDeviceSynchronize());
    
    // 将结果拷贝回主机
    CHECK(cudaMemcpy(h_output.data(), d_output, total_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    
    // 获取计时结果
    uint64_t h_clock;
    CHECK(cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool valid = true;
    for (size_t i = 0; i < total_elements; ++i) {
        if (h_output[i] != h_input[i]) {
            valid = false;
            std::cerr << "验证失败 at index " << i << ": 预期 " << h_input[i] 
                      << ", 实际 " << h_output[i] << std::endl;
            break;
        }
    }
    
    if (valid) {
        std::cout << "数据拷贝验证成功!" << std::endl;
        std::cout << "总耗时: " << h_clock << " GPU时钟周期" << std::endl;
    }
    
    // 释放资源
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_clock));
    
    return EXIT_SUCCESS;
}
```
### 异步加载
![](asserts/Pasted%20image%2020250822092134.png)
```cpp
// 主机端调用示例（简化，需补充完整错误检查、内存管理等）
int main() {
    const size_t total_elements = 1024;
    const size_t copy_count = 2;
    const size_t block_dim = 256;
    const size_t grid_dim = (total_elements + block_dim * copy_count - 1) / (block_dim * copy_count);
    const size_t shared_mem_size = block_dim * copy_count * sizeof(float);  // 假设 T 是 float

    float *d_global, *d_output;
    uint64_t *d_clock;
    cudaMalloc(&d_global, total_elements * sizeof(float));
    cudaMalloc(&d_output, total_elements * sizeof(float));
    cudaMalloc(&d_clock, sizeof(uint64_t));

    // 主机端准备数据并拷贝到设备...

    async_copy_kernel<float><<<grid_dim, block_dim, shared_mem_size>>>(
        d_global, 
        d_output, 
        d_clock, 
        copy_count, 
        total_elements
    );

    cudaDeviceSynchronize();
    // 后续处理结果、计时数据...
    return 0;
}
```


![](asserts/Pasted%20image%2020250822092356.png)


![](asserts/Pasted%20image%2020250822092517.png)



![](asserts/Pasted%20image%2020250822092619.png)




![](asserts/Pasted%20image%2020250822092627.png)


![](asserts/Pasted%20image%2020250822092658.png)

![](asserts/Pasted%20image%2020250822092751.png)


![](asserts/Pasted%20image%2020250822092829.png)

![](asserts/Pasted%20image%2020250822092918.png)



![](asserts/Pasted%20image%2020250822093022.png)

![](asserts/Pasted%20image%2020250822093031.png)

![](asserts/Pasted%20image%2020250822093129.png)

![](asserts/Pasted%20image%2020250822093309.png)




![](asserts/Pasted%20image%2020250822093318.png)


![](asserts/Pasted%20image%2020250822093358.png)





![](asserts/Pasted%20image%2020250822093458.png)



![](asserts/Pasted%20image%2020250822093526.png)


![](asserts/Pasted%20image%2020250822093608.png)


![](asserts/Pasted%20image%2020250822093617.png)





![](asserts/Pasted%20image%2020250822093708.png)



![](asserts/Pasted%20image%2020250822093732.png)


![](asserts/Pasted%20image%2020250822093926.png)


![](asserts/Pasted%20image%2020250822100536.png)

![](asserts/Pasted%20image%2020250822100606.png)
- 获得指向设备测的指针
![](asserts/Pasted%20image%2020250822100655.png)


![](asserts/Pasted%20image%2020250822100724.png)



![](asserts/Pasted%20image%2020250822100752.png)

![](asserts/Pasted%20image%2020250822100820.png)


![](asserts/Pasted%20image%2020250822100831.png)

![](asserts/Pasted%20image%2020250822100949.png)



![](asserts/Pasted%20image%2020250822101135.png)

![](asserts/Pasted%20image%2020250822101142.png)

![](asserts/Pasted%20image%2020250822101250.png)


![](asserts/Pasted%20image%2020250822101301.png)




![](asserts/Pasted%20image%2020250822101358.png)
### CUDA Graph
![](asserts/Pasted%20image%2020250822101612.png)


![](asserts/Pasted%20image%2020250822101623.png)
#### 具体实现
![](asserts/Pasted%20image%2020250822101654.png)




![](asserts/Pasted%20image%2020250822101743.png)

![](asserts/Pasted%20image%2020250822101819.png)
核函数的发射和调用和cpu有关

![](asserts/Pasted%20image%2020250822101935.png)

![](asserts/Pasted%20image%2020250822101951.png)




![](asserts/Pasted%20image%2020250822102001.png)

### 并行系统优化
![](asserts/Pasted%20image%2020250822102049.png)

![](asserts/Pasted%20image%2020250822102149.png)

![](asserts/Pasted%20image%2020250822102312.png)















