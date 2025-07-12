WORKDIR /notebooks
### 作用

- 设置 **当前工作目录**（working directory）为容器内的 `/notebooks` 目录。
- 后续所有的 `RUN`、`CMD`、`ENTRYPOINT`、`COPY`、`ADD` 等命令默认都在这个目录下执行。

### 原理

- 如果 `/notebooks` 目录不存在，Docker 会自动创建它。
- 这样保证你的容器内环境有一个固定且干净的工作路径，方便管理文件和运行程序。
    

### 使用场景

- 规范容器文件操作目录，避免路径混乱。
- 在 Jupyter 容器中，通常将代码或笔记本放在固定目录，便于挂载和访问。