### 1. 早期 $\text{System V}$ 的问题（混淆）

- **标志：** 使用 $\text{O\_NDELAY}$ 设置非阻塞模式。
    
- **返回值：** 如果非阻塞 $\text{read}$ **无数据可读**，它返回 **0**。
    
- **混淆：** $\text{read}$ 返回 **0** 在 Unix 系统中**通常表示文件结束**（EOF, End-Of-File）。这使得程序无法区分：当前是**无数据可读（暂时阻塞）**还是**文件已经读完（永久结束）**。
    

### 2. $\text{POSIX.1}$ 的标准化解决方案（清晰）

- **标志：** 使用 $\text{O\_NONBLOCK}$ 设置非阻塞模式。
    
- **返回值：** $\text{POSIX.1}$ 标准要求，对于一个非阻塞描述符，如果 $\text{read}$ **无数据可读**，它必须返回 **$-1$**，并将全局错误变量 $\text{errno}$ 设置为 **$\text{EAGAIN}$**（或 $\text{EWOULDBLOCK}$）。
    

### 3. 现代实践建议

- $\text{O\_NDELAY}$ 仅用于向后兼容，**不应在新应用程序中使用**。
    
- 应**只使用** $\text{POSIX.1}$ 规定的 **$\text{O\_NONBLOCK}$** 标志和 **$\text{-1}$ ($\text{EAGAIN}$)** 的返回值特征，以确保代码的清晰性和可移植性。
    

**总而言之，这段话解释了 $\text{EINTR}$ 之外的另一种常见 $\text{read}$ 返回 $-1$ 的情况：**

- $\text{read}$ 返回 $-1$ 且 $\text{errno} = \text{EINTR}$：**操作被信号中断**，应重试（除非是终止信号）。
    
- $\text{read}$ 返回 $-1$ 且 $\text{errno} = \text{EAGAIN/EWOULDBLOCK}$：**描述符处于非阻塞模式，当前无数据可读**，应稍后重试。
    
- $\text{read}$ 返回 $\geq 0$ 的值：
    
    - **$\text{N} > 0$**：成功读取了 $\text{N}$ 个字节。
        
    - **$\text{N} = 0$**：文件描述符已到达**文件尾端**（EOF）。