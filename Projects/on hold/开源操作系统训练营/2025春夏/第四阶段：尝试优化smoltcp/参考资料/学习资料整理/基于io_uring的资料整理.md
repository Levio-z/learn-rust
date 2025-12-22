
### 1. Async/Await 及运行时设计
- [通过迭代 WebServer 逐步深入 Rust 异步编程](https://blog.windeye.top/rust_async/learningrustasyncwithwebserver/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc
-  [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc)
- [RFC 2592: futures](https://rust-lang.github.io/rfcs/2592-futures.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc)
- ([200行代码讲透RUST FUTURES](https://stevenbai.top/rust/futures_explained_in_200_lines_of_rust/?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc) & [英文版](https://web.archive.org/web/20230203001355/https://cfsamson.github.io/books-futures-explained/introduction.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX98Xprf1JF8HOn9W5ouCMDnokWpUOOGtp1pRA3dqmc))
- [https://os.phil-opp.com/async-await/](https://os.phil-opp.com/async-await/)经典 Rust OS 教程中异步章节，帮助理解 async/await 在内核环境中的实现原理。
	- 6小时
- [Rust Runtime 设计与实现-科普篇](https://www.ihcblog.com/rust-runtime-design-1/)（monoio 作者写作，深入浅出讲解 Rust 异步运行时设计）
	- [https://www.ihcblog.com/rust-runtime-design-2/](https://www.ihcblog.com/rust-runtime-design-2/)
- [Zaid Humayun’s Blog: Async Runtimes Part II](https://redixhumayun.github.io/async/2024/09/18/async-runtimes-part-ii.html) 和 [Concurrency](https://redixhumayun.github.io/concurrency/) 系列，专业深度讲解 async runtime 内部机制。
- https://zhuanlan.zhihu.com/p/92679351
- [https://web.archive.org/web/20230324130904/https://cfsamson.github.io/books-futures-explained/](https://web.archive.org/web/20230324130904/https://cfsamson.github.io/books-futures-explained/)Futures Explained in 200 Lines of Rust:
- [https://web.archive.org/web/20220529000219/https://cfsamson.gitbook.io/green-threads-explained-in-200-lines-of-rust/](https://web.archive.org/web/20220529000219/https://cfsamson.gitbook.io/green-threads-explained-in-200-lines-of-rust/)Green Threads Explained in 200 Lines of Rust:
- 异步运行时设计[https://toetoe55.github.io/async-rt-book/](https://toetoe55.github.io/async-rt-book/)
### 2. 协程与绿色线程基础
理论与代码实现
- [无栈协程与有栈协程比较](https://mthli.xyz/stackful-stackless/)
-  [Matthew Lee: C语言实现协程](https://mthli.xyz/coroutines-in-c/)
- 无栈协程: [https://mthli.xyz/coroutines-in-c/](https://mthli.xyz/coroutines-in-c/)
- rCore-Tutorial-v3 中绿色线程协程示例代码：
    - [stackless_coroutine.rs](https://github.com/rcore-os/rCore-Tutorial-v3/blob/main/user/src/bin/stackless_coroutine.rs)    
    - [stackful_coroutine.rs](https://github.com/rcore-os/rCore-Tutorial-v3/blob/main/user/src/bin/stackful_coroutine.rs)
-  rust无锁链表 [代码杂谈：无锁编程 - Rust精选](https://rustmagazine.github.io/rust_magazine_2021/chapter_12/lock-free.html)
- [深入理解 Linux 的 epoll 机制](https://mp.weixin.qq.com/s?__biz=MzU0OTE4MzYzMw==&mid=2247515011&idx=2&sn=3812f80dd80bb27340d5849df8d1cec0&chksm=fbb1327dccc6bb6bfd5ab7f9da23220ade44e88e2f8d2506b7e0868bb84665a95f026eddb82d&scene=27)、 [深入揭秘epoll的底层原理、内核源码实现及常见问题 - 知乎](https://zhuanlan.zhihu.com/p/9584765213)


### 3.io_uring 及 Linux 异步 I/O

**官方文档与原理理解**
- - [Linux I/O 栈与零拷贝技术全揭秘|Strike Freedom](https://strikefreedom.top/archives/linux-io-stack-and-zero-copy#io-%E5%A4%9A%E8%B7%AF%E5%A4%8D%E7%94%A8%E5%92%8C%E5%BC%82%E6%AD%A5-io-epoll--io_uring)
- [io_uring 官方文档（PDF）](https://kernel.dk/io_uring.pdf)（核心底层原理、接口设计、提交与完成队列机制）
	- 通过这个文档比较系统的了解了io uring 的背景和、原理和用法。
- [Linux 异步 I/O 框架 io_uring：基本原理、程序示例与性能压测（2020）](https://arthurchiao.art/blog/intro-to-io-uring-zh)
- 关于 io_uring 的论文 ([链接](https://kernel.dk/io_uring.pdf), [翻译](https://icebergu.com/archives/linux-iouring))
- [【译】高性能异步 IO —— io_uring (Effecient IO with io_uring) - Iceber Gu Blog](https://icebergu.com/archives/linux-iouring)
**Rust 生态相关实现**
- https://zhuanlan.zhihu.com/p/346219893
- [io-uring Rust crate](https://docs.rs/io-uring/latest/io_uring/) （官方封装库）
	- [arthurchiao 的 io_uring 介绍博客](https://arthurchiao.art/blog/intro-to-io-uring-zh/)（翻译和示例，重点在性能分析）
	- tokio 官方封装的 io uring 库。我的异步运行时也基于这个库。
	
- [tokio-uring](https://github.com/tokio-rs/tokio-uring)（tokio 官方基于 io_uring 的异步运行时实现）
	- tokio 官方实现的基于 uring io 的异步运行时，在自己的异步运行时的时候接口和一些比较棘手的问题有大量参考里面的实现方式。
- [https://arthurchiao.art/blog/intro-to-io-uring-zh/](https://arthurchiao.art/blog/intro-to-io-uring-zh/)

[字节 monoio 的一个介绍文档](https://rustmagazine.github.io/rust_magazine_2021/chapter_12/monoio.html)
- 参考这个文档，对基于 uring io 的异步运行时实现上可能遇到的问题和解决方案有了一个大概了解。

**参考**
- [https://zia6.github.io/2024/12/22/io-uring%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/](https://zia6.github.io/2024/12/22/io-uring%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/)

### 4. smol 生态与 tokio 源码剖析 异步运行时的设计
- 是一个非常知名的、从头用 Rust 编写的、面向嵌入式和裸机环境的 TCP/IP 协议栈。它的设计目标就是轻量级、`no_std`，并且与 Rust 的异步生态（`async/await`）有良好的集成。
- smol async runtime 深入分析博客：[https://systemxlabs.github.io/blog/smol-async-runtime/](https://systemxlabs.github.io/blog/smol-async-runtime/)
- 重点理解 smol 的 polling / async-io / async-task / async-executor 模块，实现了 reactor、executor、task 驱动机制的核心。
- uring_io 在 smol 中实现相关的 issue
	- [https://github.com/smol-rs/async-fs/issues/24](https://github.com/smol-rs/async-fs/issues/24)
	- [https://github.com/smol-rs/async-io/issues/39](https://github.com/smol-rs/async-io/issues/39)
- [async-io PR:Integrate io_uring in the Reactor](https://github.com/smol-rs/async-io/pull/85)smol 中实现基于 uring io 的可能性



- [Tokio 源码分析记录](https://docs.qq.com/doc/DTGxhd3h5QnVFdVhZ)，包括 task 调度和 Schedule 的多线程细节。
- 结合 [tony612 的 Tokio-internals](https://tony612.github.io/tokio-internals/)博客，系统梳理 tokio 运行时设计。
- [https://github.com/smol-rs/smol](https://github.com/smol-rs/smol)smol源码
- tokio阅读
	-  [https://zia6.github.io/2024/12/10/Tokio-Task%E9%98%85%E8%AF%BB/](https://zia6.github.io/2024/12/10/Tokio-Task%E9%98%85%E8%AF%BB/)
	- [https://zia6.github.io/2024/12/10/Tokio-Task%E9%98%85%E8%AF%BB/](https://zia6.github.io/2024/12/10/Tokio-Task%E9%98%85%E8%AF%BB/)多线程下的Schedule
- - [字节跳动 | Rust 异步运行时的设计与实现 - Rust精选](https://rustmagazine.github.io/rust_magazine_2021/chapter_12/monoio.html)
### 5. 自己实现异步运行时
- io-uring 的异步运行时
	1. [https://github.com/bytedance/monoio](https://github.com/bytedance/monoio) （字节跳动）
	2. [https://github.com/DataDog/glommio](https://github.com/DataDog/glommio)
	3. [https://github.com/tokio-rs/tokio-uring](https://github.com/tokio-rs/tokio-uring)
	4. [https://github.com/compio-rs/compio](https://github.com/compio-rs/compio) （清华 王yuyi）
	5. [https://github.com/cmazakas/fiona-rs](https://github.com/cmazakas/fiona-rs) (使用自己的 axboe-liburing)
	6. [https://github.com/KuiBaDB/kbio](https://github.com/KuiBaDB/kbio) （旧）
	7. [https://github.com/r58Playz/async-uring](https://github.com/r58Playz/async-uring)
- 参考
	- [我的async-fs-uring](https://github.com/yonchicy/async-fs-uring/tree/master)
		- 实现了一个建议的基于 reactor 模式异步运行时，并实现了基于uring io的文件读。主要思想就是把 io 委托给 reactor 去提交，然后 reactor 不断轮询，如果有 io 完成了，就返回给对应的异步任务。实现过程中比较困难的点就是buf 管理，需要保证 buf 在异步读过程中一直有效。我这里做法是直接把 buf 的所有权移交给 UringReadFuture.这只是一个权宜之计，因为我这里实现的比较简单，在异步读进行过程中 UringReadFuture不会被 drop 掉。实际上后来也阅读了 tokio-uring 的相关设计文档，也了解到了一些更合理的设计方案，但是还没有时间来实现。[blog 的链接](https://github.com/rcore-os/blog/pull/663/commits/24126fa945bc2fc766e616a99e3dfde811138bd1)
	- [https://github.com/zjp-CN/os-notes/tree/main/async-runtime/async-rt-02-io-uring](https://github.com/zjp-CN/os-notes/tree/main/async-runtime/async-rt-02-io-uring)
	- [https://github.com/ZIYAN137/my_runtime.git](https://github.com/ZIYAN137/my_runtime.git)
	- [https://github.com/moyigeek/simple-rust-async](https://github.com/moyigeek/simple-rust-async)
	- 异步运行时commits:
		- master 分支的reactor + io_uring 实现：
			- [https://github.com/reganzm/aio/commits/main/](https://github.com/reganzm/aio/commits/main/)
		- reactor_epoll分支的reactor+epoll实现：
			- [https://github.com/reganzm/aio/commits/reactor_epoll](https://github.com/reganzm/aio/commits/reactor_epoll)
		- 支持异步输入的os:[https://github.com/reganzm/aos](https://github.com/reganzm/aos)
		- 支持异步的os commits:[https://github.com/reganzm/aos/commits/main/](https://github.com/reganzm/aos/commits/main/)
	-  reganzm/aio：[https://github.com/reganzm/aio](https://github.com/reganzm/aio) （基于 io_uring 的异步 runtime）
	-  simple-rust-async：[https://github.com/moyigeek/simple-rust-async](https://github.com/moyigeek/simple-rust-async) （简易版 async runtime，理解 async 基础）
- 总结
	- [https://github.com/rcore-os/blog/pull/631](https://github.com/rcore-os/blog/pull/631)
	- [git@github.com:a6d9a6m/runtime-io_uring.git](https://docs.qq.com/doc/git@github.com:a6d9a6m/runtime-io_uring.git)
	- [https://github.com/rcore-os/blog/pull/659/](https://github.com/rcore-os/blog/pull/659/)

### 6.异步操作系统
- [https://gitlab.eduxiji.net/educg-group-26010-2376550/T202418123993075-2940](https://gitlab.eduxiji.net/educg-group-26010-2376550/T202418123993075-2940)Phoenix 的实现
- [https://gitlab.eduxiji.net/PLNTRY/OSKernel2023-umi](https://gitlab.eduxiji.net/PLNTRY/OSKernel2023-umi)
### 7.额外参考资料

- Rust Pin 概念与底层原理博客合集
    - [https://folyd.com/blog/rust-pin-advanced/](https://folyd.com/blog/rust-pin-advanced/)
    - [https://blog.cloudflare.com/pin-and-unpin-in-rust/](https://blog.cloudflare.com/pin-and-unpin-in-rust/)
- Glommio 异步运行时设计博客：[https://www.ihcblog.com/rust-runtime-design-2/](https://www.ihcblog.com/rust-runtime-design-2/)
    
- 训练营群友 smol 博客和 issue：
    
    - [https://github.com/smol-rs/async-fs/issues/24](https://github.com/smol-rs/async-fs/issues/24)
        
    - [https://github.com/smol-rs/async-io/issues/39](https://github.com/smol-rs/async-io/issues/39)
- 《[Implementing async APIs for microcontroller peripherals](https://beaurivage.io/atsamd-hal-async/)》[atsamd-hal](https://github.com/atsamd-rs/atsamd)维护者介绍 atsamd 库（微控制器外设异步库）的核心设计。
- trap的思路：[https://github.com/AsyncModules/async-os/blob/main/modules/trampoline/trap.md](https://github.com/AsyncModules/async-os/blob/main/modules/trampoline/trap.md)
1. 异步的网络协议栈的实现：
	1. axnet 的 async 版本：[https://github.com/AsyncModules/async-os/tree/main/modules/async_net](https://github.com/AsyncModules/async-os/tree/main/modules/async_net)
	2. embassy_net：[https://github.com/embassy-rs/embassy/tree/main/embassy-net](https://github.com/embassy-rs/embassy/tree/main/embassy-net)

2. Ariel-OS
	1. 仓库：[https://github.com/ariel-os/ariel-os](https://github.com/ariel-os/ariel-os)
	2. 作者博客：[Building a CoAP application on Ariel OS](https://christian.amsuess.com/blog/website/2025-03-27_ariel_coap/)
	3. 背景：
	
	4. RustWeek 演讲: [Ariel OS - An Open Source Embedded Rust OS for Networked Multi-Core Microcontrollers](https://rustweek.org/talks/emmanuel/)
	5. 论文《[RIOT-ML: toolkit for over-the-air secure updates and performance evaluation of TinyML models](https://link.springer.com/content/pdf/10.1007/s12243-024-01041-5.pdf)》| [RIOT-ML 仓库](http://github.com/TinyPART/RIOT-ML)
	6. [《关于 Ariel-OS 和嵌入式 Rust 的论文（硕士论文等）的可选主题列表》](https://github.com/ariel-os/ariel-os/wiki/Thesis-Subjects)
	
	7. DeepWiki: [https://deepwiki.com/ariel-os/ariel-os](https://deepwiki.com/ariel-os/ariel-os)

3. embassy
	1. 仓库：[https://github.com/embassy-rs/embassy](https://github.com/embassy-rs/embassy)
	2. [embassy-executor 的 integrated-timers 和任务机制](https://docs.qq.com/doc/DTG1WWGRReXZ4V3NG)
	3. [embassy 使用记录之 TaskStorage](https://zjp-cn.github.io/os-notes/embassy-task.html)
	4. [embassy_time_driver::Driver](https://zjp-cn.github.io/os-notes/embassy-timer.html)
	5. [embassy Sync / Channel](https://zjp-cn.github.io/os-notes/embassy-sync.html)
	6. 刘同学的笔记：[async-summary](https://github.com/liu0fanyi/async-summary) (需要安装 treesheets 软件查看文件)
	7. [embassy_preempt：基于Rust异步机制的嵌入式操作系统调度模块](https://www.yuque.com/xyong-9fuoz/hg8kgr/culbvrzfn9qu9lby) by 袁子为、施诺晖
	8. DeepWiki: [https://deepwiki.com/embassy-rs/embassy](https://deepwiki.com/embassy-rs/embassy)


- 200行绿色线程作者的书: [https://github.com/PacktPublishing/Asynchronous-Programming-in-Rust](https://github.com/PacktPublishing/Asynchronous-Programming-in-Rust)
- - 嵌入式学习的基础推荐书：[https://www.amazon.com/Making-Embedded-Systems-Patterns-Software/dp/1449302149](https://www.amazon.com/Making-Embedded-Systems-Patterns-Software/dp/1449302149)

- Rust Atomics and Locks ([链接](https://marabos.nl/atomics/))
---

### 8、总结建议与拓展知识点

1. **从理论到实践分层推进**
    
    - 先理解 async/await、Future trait 及 Pin 机制（Rust 异步编程基础）
        
    - 学习 reactor 模型（如 smol）与多线程 executor（如 tokio）
        
    - 深入 io_uring 机制，理解内核异步接口
        
    - 综合设计自己的轻量异步 runtime，实现微内核级异步操作系统或 hypervisor。
        
2. **关注运行时设计中的难点**
    
    - 任务调度（Schedule）
        
    - 异步 I/O 多路复用（epoll/io_uring）
        
    - 任务唤醒机制（Waker/Context）
        
    - Pinning 及内存安全
        
3. **结合绿色线程和协程实现，强化异步概念**
    
    - 绿色线程底层切换机制和栈管理
        
    - async/await 语法糖对协程的转换
        
4. **阅读源码时注意对照架构图与调用链，理解设计动机**
    
    - 多线程并发模型
        
    - Reactor-Executor 的分工
        
    - 资源竞争与同步问题
        
