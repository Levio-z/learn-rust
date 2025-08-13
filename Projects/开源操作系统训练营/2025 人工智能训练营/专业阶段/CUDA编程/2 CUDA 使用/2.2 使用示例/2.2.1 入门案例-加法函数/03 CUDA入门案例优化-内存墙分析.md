![](asserts/Pasted%20image%2020250812150817.png)

![](asserts/Pasted%20image%2020250812150847.png)

![](asserts/Pasted%20image%2020250812150831.png)


![](asserts/Pasted%20image%2020250812150915.png)

- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf

- GPU内存带宽


### 计算
[内存墙概览](../Atomic%20Notes/内存墙/内存墙概览.md)
示例： 
◆ Float 加法: 1 𝐹𝐿𝑂𝑃 /12 𝑏𝑦𝑡𝑒𝑠 = 1 /12 ≈ 𝟎. 𝟎𝟖𝟑 𝑭𝑳𝑶𝑷 𝒃𝒚𝒕𝒆 
◆ A100： 
◆ 19.5 TFLOPS = 𝟏. 𝟗𝟓 × 𝟏𝟎𝟏𝟑 𝒐𝒑/𝒔𝒆𝒄 
◆ 2039 GB/s = 𝟐. 𝟎𝟑𝟗 × 𝟏𝟎𝟏𝟐 𝒃𝒚𝒕𝒆𝒔/𝒔𝒆𝒄 
◆ 𝐼𝑚𝑎𝑥 = 𝟏.𝟗𝟓×𝟏𝟎𝟏𝟑 𝑭𝑳𝑶𝑷/𝒔𝒆𝒄 𝟐.𝟎𝟑𝟗×𝟏𝟎𝟏𝟐 𝒃𝒚𝒕𝒆𝒔/s𝒆𝒄 ≈ 𝟗. 𝟓𝟔 𝑭𝑳𝑶𝑷 𝒃𝒚𝒕𝒆 

◆ 确实是访存密集型
◆ 加法是比较典型的访存密集型算子 
[02 CUDA性能分析工具 Nsight Compute](2%20CUDA%20使用/2.3%20常见问题与优化/2.3.1%20性能分析工具/02%20CUDA性能分析工具%20Nsight%20Compute.md)
### Nsight Compute 分析
![](asserts/Pasted%20image%2020250812160203.png)
每个核函数一行 
- 包含函数名、GPU 运行时间、计算吞吐%、内存吞吐% 等信息
- %达到最大性能的多少
![](asserts/Pasted%20image%2020250812160227.png)
Details → GPU Speed Of Light Throughput
- 基础强度0.12
![](asserts/Pasted%20image%2020250812160624.png)

观察： 
◆ 内存吞吐利用率比计算高不少 → 访存密集型 
◆ ncu 会根据测量的指标给出建议

加法的性能瓶颈在内存访问上 
◆ 因此想办法：提升其访存效率 观察： 
◆ 内存使用率虽相对较高，但仍有较大提升空间 
◆ 要增大内存带宽使用，如果一次多传输数据呢？…

