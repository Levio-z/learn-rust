```
file target/riscv64gc-unknown-none-elf/release/os

target/riscv64gc-unknown-none-elf/release/os: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV), statically linked, not stripped
```

我们以 `release` 模式生成了内核可执行文件，它的位置在 `os/target/riscv64gc.../release/os` 。接着我们通过 `file` 工具查看它的属性，可以看到它是一个运行在 64 位 RISC-V 架构计算机上的可执行文件，它是静态链接得到的。