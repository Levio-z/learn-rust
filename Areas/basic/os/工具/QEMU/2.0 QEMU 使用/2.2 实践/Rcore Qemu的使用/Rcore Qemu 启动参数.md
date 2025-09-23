

- 需要在怎样的环境里面运行我们编写的内核？

```
run-inner: build
	@qemu-system-riscv64 \\
		-machine virt \\
		-nographic \\
		-bios $(BOOTLOADER) \\
		-device loader,file=$(KERNEL_BIN),addr=$(KERNEL_ENTRY_PA)
```

- `@qemu-system-riscv64`
    - 调用 QEMU 的 RISC-V 64 位系统仿真器，它包含CPU 、物理内存以及若干 I/O 外设。
- `machine virt`
    - 使用 QEMU 提供的 **虚拟通用 RISC-V 开发板**。
        - 我们知道，即使同属同一种指令集架构，也会有很多种不同的计算机配置，比如 CPU 的生产厂商和型号不同，支持的 I/O 外设种类也不同。关于 `virt` 平台的更多信息可以参考 [1](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter1/3first-instruction-in-kernel1.html#virt-platform) 。Qemu 还支持模拟其他 RISC-V 计算机，其中包括由 SiFive 公司生产的著名的 HiFive Unleashed 开发板。
- `nographic`
    - 禁用图形界面。
- `bios $(BOOTLOADER)`
    - 指定 QEMU 启动时加载的 BIOS 映像（Bootloader 作用）。
    - **对于 RustSBI 裸机系统，`$(BOOTLOADER)` 常为 RustSBI 提供的 `rustsbi-qemu.bin`**。
        - 这里我们使用预编译好的 `rustsbi-qemu.bin` ，它需要被放在与 `os` 同级的 `bootloader` 目录下，该目录可以从每一章的代码分支中获得。
    - 替代默认 BIOS，从而实现 **定制化启动流程**（如进入 Supervisor Mode，加载 payload）。
- `-device loader,file=$(KERNEL_BIN),addr=$(KERNEL_ENTRY_PA)`
    - 这一句用于将你的操作系统内核（KERNEL）手动加载到指定物理地址。
        - 参数说明：
            - `device loader,...`：使用 QEMU 的 `loader` 设备模拟器，向内存直接加载内容。
            - `file=$(KERNEL_BIN)`：指定要加载的内核 ELF 或 BIN 文件。
            - `addr=$(KERNEL_ENTRY_PA)`：指定要将该文件映射到的物理地址（PA = Physical Address）。
        - 这里我们载入的 `os.bin` 被称为 **内核镜像** ，它会被载入到 Qemu 模拟器内存的 `0x80200000` 地址处。 那么内核镜像 `os.bin` 是怎么来的呢？上一节中我们移除标准库依赖后会得到一个内核可执行文件 `os` ，将其进一步处理就能得到 `os.bin` ，具体处理流程我们会在后面深入讨论。
    - 例子
        `KERNEL_BIN = target/riscv64gc-unknown-none-elf/release/os KERNEL_ENTRY_PA = 0x80200000`
        
- QEMU 启动逻辑图解
    ```jsx
    QEMU (riscv64)
     ├── BIOS: rustsbi-qemu.bin (via -bios)
     │    └── Boot logic: 启动内核 at 0x80200000
     └── 内核: os.bin (via -device loader, addr=0x80200000)
          └── Rust OS 的入口函数
    ```
    
- 其他的参数

```jsx
-M 指定内存，默认为128M（所以如果见到rcore计算出来的内核物理地址上限请记住这个值）
-SMP指定核心数
-netdev来指定网络设备相关参数
```

