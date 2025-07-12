- Master进程，被称为coordinator协调器，负责orchestrate编排wokers，把map jobs分配给它们
- reduce、map被称为task任务

1. coordinator协调器将文件分配给特定的workers，worker对分配到的文件调用map函数
    
2. worker将执行map函数产生的中间结果存储到本地磁盘
    
3. worker的map函数执行完毕后并告知master中间结果存储的位置
    
4. 所有worker的map执行完毕后，coordinator协调器分配worker执行reduce函数
    
5. worker记录分配到的map中间结果，获取数据，按key键sort排序，在每个key、values集合上调用reduce函数
    
6. 每个reduce函数执行时产生结果数据，你可以聚合输出文件获取最终结果
    

​ 输入文件在全局文件系统中，被称为GFS。Google现在使用的是不同的global file system，但该论文中使用的是GFS。

​ 上面流程最后reduce输出结果会被保存到GFS，而map产生的中间文件不会被保存到GFS中（而是保存到worker运行的本地机器上）。

---

问题：在远程读取进程中，文件是否会传输到reducer？

回答：是的。map函数产生的中间结果存放在执行map函数的worker机器的磁盘上，而之后解调器分配文件给reducer执行reduce函数时，中间结果数据需要通过网络传输到reducer机器上。这里其实很少有网络通信，因为一个worker在一台机器上，而每台机器同时运行着worker进程和GFS进程。worker运行map产生中间结果存储在本地，而之后协调器给worker分配文件以执行reduce函数时，才需要通过网络获取中间结果数据，最后reduce处理完在写入GFS，写入GFS的动作也往往需要通络传输。

**问题：协调器是否负责对数据进行分区，并将数据分发到每个worker或机器上？**

**回答：不是的。mapreduce运行用户程序，这些输入数据在GFS中。（也就是说协调器告知worker从GFS取哪些数据进行map，后续协调器又告知worker从哪些worker机器上获取中间结果数据进行reduce，最后又统一写入到GFS中）**

问题：这里涉及的排序是如何工作的？比如谁负责排序，如何排序？

回答：在中间结果数据传递到reduce函数之前，mapreduce库进行一些排序。比如所有的中间结果键a、b、c到一个worker。比如`(a,1) (b,1) (c,1) (a,1)` 数据，被排序成`(a,1) (a,1) (b,1) (c,1)` 后才传递给reduce函数。

问题：很多函数式编程是否可以归结为mapreduce问题？

回答：是的。因为map、reduce函数的概念，在函数式编程语言中非常常见，或者说函数式编程真是map、reduce的灵感来源。