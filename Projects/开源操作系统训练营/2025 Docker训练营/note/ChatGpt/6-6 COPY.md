### 作用

- 将 **构建上下文**（Docker build 时所在目录）中的文件 `sample-notebook.ipynb` 复制到当前工作目录（即 `/notebooks`，因为前面设置了 `WORKDIR`）内。
    

### 具体说明

- `.` 代表当前工作目录，即 `/notebooks`。
    
- 文件复制操作是从宿主机到镜像文件系统。
    
- 复制完成后，你的容器内 `/notebooks` 目录下会有 `sample-notebook.ipynb` 文件。
    

### 使用场景

- 将本地的 Jupyter Notebook 文件放入容器，方便启动时即加载示例笔记。
    
- 实现“代码 + 数据 +配置”一起打包，方便分发和运行。