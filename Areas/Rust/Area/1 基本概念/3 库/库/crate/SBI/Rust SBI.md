事实上，作为对我们之前提到的 [应用程序执行环境](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter1/1app-ee-platform.html#app-software-stack) 的细化，RustSBI 介于底层硬件和内核之间，是我们内核的底层执行环境。

之前我们对 RustSBI 的了解仅限于它会在计算机启动时进行它所负责的环境初始化工作，并将计算机控制权移交给内核。但实际上作为内核的执行环境，它还有另一项职责：**即在内核运行时响应内核的请求为内核提供服务。当内核发出请求时，计算机会转由 RustSBI 控制来响应内核的请求，待请求处理完毕后，计算机控制权会被交还给内核**。从内存布局的角度来思考，**每一层执行环境（或称软件栈）都对应到内存中的一段代码和数据，这里的控制权转移指的是 CPU 从执行一层软件的代码到执行另一层软件的代码的过程**。这个过程与我们使用高级语言编程时调用库函数比较类似。

这里展开介绍一些相关术语：从第二章将要讲到的 [RISC-V 特权级架构](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter2/1rv-privilege.html#riscv-priv-arch) 的视角来看，我们编写的 OS 内核位于 Supervisor 特权级，而 RustSBI 位于 Machine 特权级，也是最高的特权级。类似 RustSBI 这样运行在 Machine 特权级的软件被称为 Supervisor Execution Environment(SEE)，即 Supervisor 执行环境。两层软件之间的接口被称为 Supervisor Binary Interface(SBI)，即 Supervisor 二进制接口。 [SBI Specification](https://github.com/riscv-non-isa/riscv-sbi-doc) （简称 SBI spec）规定了 SBI 接口层要包含哪些功能，该标准由 RISC-V 开源社区维护。RustSBI 按照 SBI spec 标准实现了需要支持的大多数功能，但 RustSBI 并不是 SBI 标准的唯一一种实现，除此之外还有社区中的前辈 OpenSBI 等等。

