RPC系统的目标：

- **使得RPC在编程使用方面和栈上本地过程调用无差异**。比如gRPC等实现，看上去Client和Server都像是在调用本地方法，对开发者隐藏了底层的网络通信环节流程

RPC semantics under failures （RPC 失败时的语义） ：

- at-least-once：至少执行一次
- at-most-once：至多执行一次，即重复请求不再处理
- Exactly-once：正好执行一次，很难做到