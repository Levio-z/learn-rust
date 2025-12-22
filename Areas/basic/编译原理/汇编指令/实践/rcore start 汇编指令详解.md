
```jsx
 # os/src/entry.asm
     .section .text.entry
     .globl _start
 _start:
     li x1, 100
```

- 实际的指令位于第 5 行，也即 `li x1, 100` 。
    - `li` 是 Load Immediate 的缩写，也即将一个立即数加载到某个寄存器，因此这条指令可以看做将寄存器 `x1` 赋值为 `100` 。
- 第 4 行我们声明了一个符号 `_start`
    - **该符号指向紧跟在符号后面的内容**——也就是位于第 5 行的指令，因此符号 `_start` 的地址即为第 5 行的指令所在的地址。
- 第 3 行我们告知编译器 `_start` 是一个全局符号，因此可以被其他目标文件使用。
- 第 2 行表明我们希望将第 2 行后面的内容全部放到一个名为 `.text.entry` 的段中。
    - 一般情况下，所有的代码都被放到一个名为 `.text` 的代码段中，这里我们将其命名为 `.text.entry` 从而区别于其他 `.text` 的目的在于我们想要确保该段被放置在相比任何其他代码段更低的地址上。这样，作为内核的入口点，这段指令才能被最先执行。

### 定义栈

```
# os/src/entry.asm
    .section .text.entry
    .globl _start
_start:
    la sp, boot_stack_top
    call rust_main

    .section .bss.stack
    .globl boot_stack_lower_bound
boot_stack_lower_bound:
    .space 4096 * 16
    .globl boot_stack_top
boot_stack_top:
```

```
- la sp, boot_stack_top
	- `la` = load address，把符号 `boot_stack_top` 的地址加载到寄存器 `sp`（stack pointer，栈指针）。
	- 作用：设置内核运行时的初始栈顶。
	- 因为栈是向下生长的，所以这里设置的是 **栈顶** 地址，而不是底部。
- call rust_main
	- 跳转到 `rust_main` 函数执行，并把返回地址保存到寄存器 `ra`。
	- `rust_main` 是用 Rust 实现的内核主函数，相当于 C 语言裸机系统中的 `main`

栈空间定义（.bss.stack 段）
```rust
.section .bss.stack
.globl boot_stack_lower_bound
boot_stack_lower_bound:
    .space 4096 * 16
.globl boot_stack_top
boot_stack_top:
```
.section .bss.stack
- 定义一个新的段，属于 **BSS 段**（未初始化的全局数据区），名字叫 `.bss.stack`。
- 在链接脚本里，`.bss.*` 通常会合并进最终的 `.bss`，因此这块区域不会存储在 ELF 文件中，而是运行时由内存清零。
boot_stack_lower_bound:
- 标签，表示栈底地址。
- 用 `globl` 导出，使得其他文件也能引用它。
.space 4096 * 16
- 在 `.bss.stack` 段分配 **4096 × 16 = 65536 (64 KiB)** 空间。
-  `.space <n>` 是汇编伪指令，用来 **在当前段分配 `n` 字节的连续空间**
- 在 `.bss` 段中使用 `.space`，这块内存 **不会在 ELF 文件中占实际存储**，运行时由内核清零。

定义符号：
-  **先导出符号，再定义符号**。
```
.globl boot_stack_lower_bound
boot_stack_lower_bound:
```

 RISC-V 架构上，栈是从高地址向低地址增长。因此，最开始的时候栈为空，栈顶和栈底位于相同的位置，我们用更高地址的符号 `boot_stack_top` 来标识栈顶的位置。同时，我们用更低地址的符号 `boot_stack_lower_bound` 来标识栈能够增长到的下限位置，它们都被设置为全局符号供其他目标文件使用。如下图所示：
![](../../../../../Projects/on%20hold/开源操作系统训练营/2025春夏/第二阶段：rcore/rCore-Turial-note/2.0%20应用程序执行环境/asserts/Pasted%20image%2020250909203114.png)
