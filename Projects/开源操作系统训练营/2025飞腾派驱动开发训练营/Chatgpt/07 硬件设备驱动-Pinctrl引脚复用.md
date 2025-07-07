![](asserts/Pasted%20image%2020250704094302.png)
## 🔧 一、Pinctrl 是什么？

> **Pinctrl**（Pin Controller）顾名思义是 **控制引脚行为的硬件/软件模块**，通常由芯片厂商提供 IP，并由 BSP 驱动控制。

**Pinctrl（Pin Controller）** 是 SoC（System on Chip）中一个专门的子系统，**负责管理 SoC 上所有物理引脚（Pins）的功能选择与电气特性配置**。它既可以是一个**硬件 IP 模块**（集成在芯片内部），也有对应的软件驱动模块（即 Linux 中的 pinctrl driver），用于完成用户空间 → 寄存器配置的映射与抽象。



- **IP（Intellectual Property）模块**：芯片内部的引脚控制逻辑电路，负责 IO MUX、上下拉、驱动电平控制等。
- **BSP（Board Support Package）**：
    - 是芯片厂商提供的低层支持代码；
    - 包括：pinctrl 驱动、设备树片段、引脚功能描述（pinmux tables）；
    - 通常你只需“声明用哪个引脚干什么”，实际配置动作由驱动 + BSP 完成。

* * *
## 📌 二、Pinctrl 的三大功能

### 1. 引脚枚举与命名（Enumerating and Naming）

* 为每个引脚赋予一个逻辑名称。
* 如 `pinA`, `pinB`，或者一组引脚组成 `pin_group`。
* 有助于驱动程序知道该使用哪个引脚。
### 2. 引脚复用（Multiplexing / IO MUX）

* **同一个引脚可以有多种功能**，如可配置为：
    * GPIO（通用输入输出）
    * I2C/SPI/UART（特定通信接口）
* **复用（Multiplexing）** 就是通过 `IO MUX` 单元，决定当前引脚连接哪个功能模块。
* 设备树指明
	* 指明使用哪些引脚
	* 复用为哪些功能

> 举例：  
> 一个引脚可以是 GPIO，但也可以是 UART_TX。Pinctrl 决定它实际连接哪条“总线”。

### 3. **引脚配置（Configuration）**

* 决定引脚的 **电气行为**：
    * 上拉 / 下拉
    * 开漏输出（Open Drain）
    * 驱动强度（Drive Strength）
    * 速度等级
    * 输入/输出方向

* * *

## 🔄 三、设备驱动中如何使用 Pinctrl？

在 Linux 驱动开发中，Pinctrl 通常由芯片厂商提供，驱动只需 **指定配置** 即可
* 指定：用到哪些引脚（如哪些 GPIO）
* 指定：这些引脚复用成什么功能（如 UART、I2C）
* 指定：这些引脚的电气特性（如是否上拉）

甚至某些简单平台在驱动中根本不需要调用 pinctrl API，**系统启动时通过 device tree（设备树）或 BSP 代码就配置好了**。
* * *

## 📷 图示说明：

右侧框图说明：
```text
引脚功能信号流 → [GPIO/I2C/UART] → IO Multiplexer（IO复用器）→ 引脚配置模块 → 连接到实际 pinA/pinB...
```

你可以理解为：
* MUX 负责“选谁来占用这个引脚”
* config 负责“占用后，这个引脚怎么工作”
* 最后输出到物理引脚 pinA~pinD
    

* * *
### [07-02 Pinctrl驱动框架](07-02%20Pinctrl驱动框架.md)

[07-03 GPIO引脚复用案例](07-03%20GPIO引脚复用案例.md)

## ✅ 总结关键词

| 术语 | 含义 |
| --- | --- |
| pinctrl | 控制引脚行为的统一模块 |
| GPIO / UART / I2C | 多种功能模块，竞争引脚使用权 |
| IO MUX | 复用选择器，决定引脚归属 |
| config | 电气配置，如上下拉等 |
| BSP | 板级支持包，通常配置这些内容 |

* * *