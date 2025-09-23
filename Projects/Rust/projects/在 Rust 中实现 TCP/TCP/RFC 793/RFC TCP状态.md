```
  
  
                              +---------+ ---------\      active OPEN
                              |  CLOSED |            \    -----------
                              +---------+<---------\   \   create TCB
                                |     ^              \   \  snd SYN
                   passive OPEN |     |   CLOSE        \   \
                   ------------ |     | ----------       \   \
                    create TCB  |     | delete TCB         \   \
                                V     |                      \   \
                              +---------+            CLOSE    |    \
                              |  LISTEN |          ---------- |     |
                              +---------+          delete TCB |     |
                   rcv SYN      |     |     SEND              |     |
                  -----------   |     |    -------            |     V
 +---------+      snd SYN,ACK  /       \   snd SYN          +---------+
 |         |<-----------------           ------------------>|         |
 |   SYN   |                    rcv SYN                     |   SYN   |
 |   RCVD  |<-----------------------------------------------|   SENT  |
 |         |                    snd ACK                     |         |
 |         |------------------           -------------------|         |
 +---------+   rcv ACK of SYN  \       /  rcv SYN,ACK       +---------+
   |           --------------   |     |   -----------
   |                  x         |     |     snd ACK
   |                            V     V
   |  CLOSE                   +---------+
   | -------                  |  ESTAB  |
   | snd FIN                  +---------+
   |                   CLOSE    |     |    rcv FIN
   V                  -------   |     |    -------
 +---------+          snd FIN  /       \   snd ACK          +---------+
 |  FIN    |<-----------------           ------------------>|  CLOSE  |
 | WAIT-1  |------------------                              |   WAIT  |
 +---------+          rcv FIN  \                            +---------+
   | rcv ACK of FIN   -------   |                            CLOSE  |
   | --------------   snd ACK   |                           ------- |
   V        x                   V                           snd FIN V
 +---------+                  +---------+                   +---------+
 |FINWAIT-2|                  | CLOSING |                   | LAST-ACK|
 +---------+                  +---------+                   +---------+
   |                rcv ACK of FIN |                 rcv ACK of FIN |
   |  rcv FIN       -------------- |    Timeout=2MSL -------------- |
   |  -------              x       V    ------------        x       V
    \ snd ACK                 +---------+delete TCB         +---------+
     ------------------------>|TIME WAIT|------------------>| CLOSED  |
                              +---------+                   +---------+
```
##### 初始阶段
- **CLOSED → LISTEN**：服务器调用 `listen()`，准备接收连接，进入被动 open。
##### 连接打开过程（三次握手）
客户端
- （1）**CLOSED → SYN_SENT**：客户端调用 `connect()` 发起主动连接，发送 SYN，创建 TCB。
- （3）**SYN_SENT → ESTABLISHED**：客户端收到 SYN+ACK → 回复 ACK → 进入 ESTABLISHED
服务器：
- （2）**LISTEN→ SYN_RCVD**：收到 SYN → 回复 SYN+ACK → 进入 SYN_RCVD
- （4）SYN_RCVD→ ESTABLISHED：收到 ACK → 进入 ESTABLISHED，此时能收到数据
##### 数据传输
- ESTABLISHED： 连接建立完毕，双向可以发送应用数据
##### 连接关闭过程（四次挥手）

客户端
- （1）**ESTABLISHED → FIN_WAIT_1**：客户端调用 `close()`，发送 FIN，进入 FIN_WAIT_1。
- （3）**FIN_WAIT_1 → FIN_WAIT_2**：收到对端 ACK，确认自己的 FIN，被动等待对方关闭。
- （5）**FIN_WAIT_2 → TIME_WAIT**：收到对端 FIN，发送 ACK，进入 TIME_WAIT，等待足够时间确保对方收到确认。
- （7）**TIME_WAIT → CLOSED**：等待 2MSL 超时后，连接彻底关闭，删除 TCB。

服务器
- （2）**ESTABLISHED → CLOSE_WAIT**：收到客户端 FIN，发送 ACK，进入 CLOSE_WAIT，等待应用关闭连接。
- （4）**CLOSE_WAIT → LAST_ACK**：应用调用 `close()`，发送 FIN，进入 LAST_ACK，等待客户端确认。
- （6）**LAST_ACK → CLOSED**：收到客户端 ACK，连接关闭，删除 TCB。

##### 异常关闭过程（CLOSING状态）

- （1）**FIN_WAIT_1 → CLOSING**：在主动关闭后（处于 FIN_WAIT_1 状态），收到对端 FIN 而不是 ACK，回复 ACK，进入 CLOSING 状态。
    
- （2）**CLOSING → TIME_WAIT**：收到对端对自己 FIN 的确认 ACK，进入 TIME_WAIT，等待确保连接安全关闭。


### 📘 **一、各状态定义与作用**

| 状态名             | 定义与作用                                            |
| --------------- | ------------------------------------------------ |
| **CLOSED**      | 初始状态，无连接存在，TCP 控制块（TCB）不存在。                      |
| **LISTEN**      | 服务端监听状态，等待客户端发起连接（`passive OPEN`）。               |
| **SYN-SENT**    | 主动连接状态，客户端发送 SYN 后，等待服务端回应（`active OPEN`）。       |
| **SYN-RCVD**    | 服务端收到 SYN 后，发送 SYN-ACK，等待客户端 ACK。                |
| **ESTABLISHED** | 双方已建立连接，可开始数据传输。                                 |
| **FIN-WAIT-1**  | 主动关闭方发送 FIN，等待对方 ACK 或 FIN。                      |
| **FIN-WAIT-2**  | 收到对方对 FIN 的 ACK，等待对方的 FIN。                       |
| **CLOSING**     | 双方几乎同时关闭，等待对方的 ACK。                              |
| **LAST-ACK**    | 被动关闭方发送 FIN，等待对方 ACK。                            |
| **TIME-WAIT**   | 主动关闭方收到 FIN 后，进入等待（2 * MSL，最大报文生存时间）以确保对方收到 ACK。 |
| **CLOSED**      | 最终状态，TCP 连接释放，TCB 被删除。                           |
