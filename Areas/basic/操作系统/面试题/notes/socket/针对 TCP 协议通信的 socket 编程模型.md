---
tags:
  - note
---
## 1. 核心观点  

| **角色**   | **步骤**      | **关键 API 函数**        | **备注**                                  |
| -------- | ----------- | -------------------- | --------------------------------------- |
| **服务器端** | 1. 初始化套接字   | `socket()`           | 创建一个监听套接字（文件描述符）。                       |
|          | 2. 绑定地址     | `bind()`             | 将套接字绑定到本地 IP 和端口。                       |
|          | 3. 开启监听     | `listen()`           | 切换为被动模式，准备接收连接。                         |
| **客户端**  | 4. 初始化套接字   | `socket()`           | 创建一个客户端套接字。                             |
| **服务器端** | **5. 等待连接** | **`accept()`**       | **阻塞**在此，等待握手完成。                        |
| **客户端**  | **6. 发起连接** | **`connect()`**      | 向服务器发起 TCP 握手。                          |
| **服务器端** | **7. 接受连接** | **`accept()`**       | **TCP 握手完成后返回**，得到一个新的**已连接套接字**（用于通信）。 |
| **双方**   | 8. 传输数据     | `read()` / `write()` | 使用新的已连接套接字进行数据交换。                       |


![](asserts/Pasted%20image%2020251018200111.png)

伪代码
```c
// 1. 创建、绑定和监听套接字（Listen_Sock_FD 是监听套接字的文件描述符）
// ... (socket, bind, listen)

while (1) { // 服务器主循环，永不停止

    printf("Server: Waiting for a new client connection...\n");

    // 2. 阻塞调用 accept()
    // 它等待连接队列中有新的连接建立完成（三次握手完成）
    // 返回的 New_Sock_FD 是用于与该客户端通信的新套接字
    New_Sock_FD = accept(Listen_Sock_FD, 
                         (struct sockaddr *)&Client_Addr, 
                         &Addr_Size);

    if (New_Sock_FD == -1) {
        perror("accept error");
        continue;
    }

    printf("Server: Successfully accepted connection from a client.\n");

    // 3. 将通信任务交给新的线程或进程处理 (多任务)
    // 这样主进程可以立即回来执行下一个 accept
    pid = fork(); 
    if (pid == 0) { // 子进程/线程负责通信
        close(Listen_Sock_FD); // 子进程不需要监听套接字
        
        // 4. 核心通信逻辑
        Handle_Client_Communication(New_Sock_FD);
        
        close(New_Sock_FD); // 通信完成后关闭已连接套接字
        exit(0);
    } else { // 父进程（主循环）
        close(New_Sock_FD); // 父进程不需要已连接套接字，只关心监听套接字
        // Loop goes back to while(1) to call accept() again
    }
}
// close(Listen_Sock_FD); // 服务器关闭时执行
```
这里需要注意的是，服务端调用 `accept` 时，连接成功了会返回一个已完成连接的 socket，后续用来传输数据。

所以，监听的 socket 和真正用来传送数据的 socket，是「**两个**」 socket，一个叫作**监听 socket**，一个叫作**已完成连接 socket**。

成功连接建立之后，双方开始通过 read 和 write 函数来读写数据，就像往一个文件流里面写东西一样。

## 2. 展开说明  

## 3. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 4. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
