```
panic!() ——> panic_handler (panic_impl)
                    │
             如果 panic=unwind 模式
                    ↓
           运行时调用 eh_personality
           实现栈展开过程，依次运行 Drop

```
[8 eh_personality 语言项详解](8%20eh_personality%20语言项详解.md)