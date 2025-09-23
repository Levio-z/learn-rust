们可以通过 **链接脚本** (Linker Script) 调整链接器的行为，使得最终生成的可执行文件的内存布局符合Qemu的预期，即内核第一条指令的地址应该位于 0x80200000 。我们修改 Cargo 的配置文件来使用我们自己的链接脚本 `os/src/linker.ld` 而非使用默认的内存布局：
```rust
 // os/.cargo/config
 [build]
 target = "riscv64gc-unknown-none-elf"

 [target.riscv64gc-unknown-none-elf]
 rustflags = [
     "-Clink-arg=-Tsrc/linker.ld", "-Cforce-frame-pointers=yes"
 ]
```

- `rustflags`
    - 这是传给 `rustc` 编译器的附加参数（相当于命令行中 `rustc <flags>`）。
- "-Clink-arg=-Tsrc/linker.ld"
    - `C`: 编译器级别的配置
    - `link-arg=...`: 给链接器传递参数（最终传给 `ld` 或 `lld`）
    - `Tsrc/linker.ld`: `T` 是链接器参数，指定使用 `src/linker.ld` 作为链接脚本
- -Cforce-frame-pointers=yes
    - 强制所有函数在栈上保留帧指针（frame pointer），即使在默认的优化级别下可能会省略它。
    - 每个函数调用在进入时会**保存调用者的帧指针（`fp`/`s0`）**，然后建立自己的栈帧：
    - 但是Release优化会
        - 使用栈指针 `sp` + 偏移量访问局部变量
        - 让 `fp/s0` 寄存器用于其它用途（比如保存变量），**增加寄存器利用率**
- **作用**：
    - 调试时可以用帧指针回溯调用栈
    - 在裸机或 OS 开发中尤其重要，因为没有操作系统提供栈展开/调试支持
- 默认情况下，优化可能会省略帧指针（frame pointer omission, FPO），加上这个选项可以保留
	- **FP 链**：显式链表，调试器可直接遍历
	- **FPO（无 FP）**：调用链和局部变量访问依赖 **DWARF 或 SP 偏移表**
	- **本质区别**：是否有显式的栈帧指针来形成调用链
	- **实践建议**：
	    - 开发裸机、内核或 no_std 时关闭 FPO
	    - 普通应用程序追求性能可开启 FPO，依赖 DWARF 调试
[案例-rcore链接脚本详解](../../../../../../basic/编译原理/链接/自定义链接脚本/案例-rcore链接脚本详解.md)
### 参考
[2.4 内核第一条指令（实践篇）](../../../../../../../Projects/开源操作系统训练营/2025春夏/第二阶段：rcore/rCore-Turial-note/2.0%20应用程序执行环境/2.4%20内核第一条指令（实践篇）.md)
