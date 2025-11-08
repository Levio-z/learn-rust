非常好的问题，这句话涉及 **信号屏蔽集（signal mask）** 与 **信号递送（signal delivery）** 的核心机制，是理解 `sigprocmask(SIG_BLOCK)` 的关键所在。  
下面分层解释原理、过程和例子。

---

### 一、信号的三种状态

在 Linux/Unix 内核中，每个进程（准确地说，每个**线程**）都维护一个**信号屏蔽集（signal mask）**。  
信号在生命周期中有三种状态：

|状态|含义|
|---|---|
|1️⃣ 未产生|信号尚未触发（比如没有调用 `kill()`）|
|2️⃣ 已产生但被阻塞|信号已到达内核，但因为被屏蔽（mask）而**暂不递送**|
|3️⃣ 已递送并处理|信号被解除屏蔽后，内核调用对应的信号处理函数（handler）|

---

### 二、`sigprocmask(SIG_BLOCK)` 的真实作用

```c
sigprocmask(SIG_BLOCK, &newmask, &oldmask);
```

意思是：

> 把 `newmask` 里的信号（这里是 `SIGUSR1` 和 `SIGUSR2`）**加入当前线程的信号屏蔽集**。  
> 即：这些信号如果被发送到该线程，不会立即递送到用户态，而是**被内核暂存（pending）**。

所以：

- 如果此时有人 `kill(getpid(), SIGUSR1)`，信号不会立刻触发 `sig_usr()`；
    
- 内核只是**记下**“有一个 SIGUSR1 待处理”；
    
- 只有当该信号被解除屏蔽（即从信号屏蔽集中移除）时，内核才会**立刻递送**它。
    

---

### 三、为什么要“暂时阻塞”

#### 背景

父子进程在 `fork()` 之后几乎同时运行，若信号太早发送（处理函数尚未安装），就会出现**竞态条件**：

> 子进程可能还没准备好等待，父进程的信号已经到达，导致“信号丢失”，子进程永远等待。

#### 解决思路

1. 先在 `TELL_WAIT()` 中 **阻塞信号**（`SIG_BLOCK`）；
    
2. 设置好信号处理函数；
    
3. 然后在真正需要等待的地方，调用：
    
    ```c
    sigsuspend(&zeromask);
    ```
    
    临时解除阻塞，让内核可以**安全地递送信号**。
    

---

### 四、`sigsuspend()` 的临时解除机制

`sigsuspend(&zeromask)` 做了三件事（原子操作）：

1. 暂时把当前信号屏蔽集替换成 `zeromask`（即空集，不屏蔽任何信号）；
    
2. 挂起当前进程，直到有信号到达；
    
3. 信号到达时内核执行对应的处理函数，然后 `sigsuspend` 返回；
    
4. 返回后信号屏蔽集恢复为调用前的状态（重新屏蔽）。
    

⚙️ **关键点**：  
这保证了等待过程**没有竞态**——解除阻塞与休眠是**原子完成**的。

---

### 五、完整的信号生命周期举例

以 `SIGUSR1` 为例，父子同步场景如下：

|步骤|操作|说明|
|---|---|---|
|1️⃣|`sigprocmask(SIG_BLOCK, {SIGUSR1, SIGUSR2}, &oldmask)`|阻塞信号，防止提前递送|
|2️⃣|`fork()`|父子都继承屏蔽状态|
|3️⃣|父进程执行 `kill(child_pid, SIGUSR1)`|信号到达，但被内核暂存（blocked）|
|4️⃣|子进程执行 `sigsuspend(&zeromask)`|临时解除阻塞并睡眠等待|
|5️⃣|内核发现有 pending 的 `SIGUSR1`，立即递送信号||
|6️⃣|信号处理函数 `sig_usr()` 执行，将 `sigflag=1`||
|7️⃣|`sigsuspend()` 被信号中断返回，恢复原屏蔽集||
|8️⃣|子进程检查 `sigflag`，继续执行逻辑||

🧠 **结论**：

> `sigprocmask(SIG_BLOCK)` 的效果是“先屏蔽信号，防止信号过早执行”；  
> `sigsuspend()` 的效果是“在合适时机临时解除屏蔽，让信号安全地执行”。

---

### 六、比喻帮助理解

可以把信号屏蔽看作一扇“防弹门”：

- `sigprocmask(SIG_BLOCK)` = 关上门：信号来了也进不来（内核暂时保存）。
    
- `sigsuspend()` = 临时开门等待信号：来了就立即处理，没来就继续等。
    
- `sigprocmask(SIG_SETMASK)` = 恢复原状态：门重新上锁，防止下次信号干扰。
    

---

### 七、总结重点

|概念|含义|
|---|---|
|信号屏蔽集|控制哪些信号暂不递送|
|`SIG_BLOCK`|将信号加入屏蔽集（阻塞）|
|`SIG_UNBLOCK`|从屏蔽集中移除（解除阻塞）|
|`sigsuspend()`|临时替换屏蔽集 + 等待信号，原子操作|
|**效果**|让信号在我们想要的时刻、且只在那个时刻被安全递送|

---

### 八、推荐练习（强化理解）

1. 写一个小实验程序：
    
    ```c
    signal(SIGUSR1, handler);
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    sigprocmask(SIG_BLOCK, &mask, NULL);
    kill(getpid(), SIGUSR1);
    printf("sent SIGUSR1, but handler not called yet\n");
    sigprocmask(SIG_UNBLOCK, &mask, NULL);
    printf("unblocked SIGUSR1\n");
    ```
    
    → 观察信号在解除阻塞后才执行 handler。
    
2. 将 `sigsuspend()` 改为 `pause()`，比较两者区别。
    
3. 在 `strace` 下观察信号的递送顺序。
    

---

### 九、底层价值（重点掌握）

高价值底层知识点：

- **信号屏蔽集机制**（sigprocmask/sigset_t）
    
- **信号递送的延迟模型**
    
- **sigsuspend 的原子解除原理**
    
- **竞态条件的消除策略**
    

掌握这部分内容能帮助你理解：

- Linux 线程的信号分发机制；
    
- 异步事件与阻塞等待模型；
    
- 系统调用层的同步与调度安全设计。
    

---

是否希望我画一个信号状态转换图（pending → blocked → delivered）来可视化解释这一过程？