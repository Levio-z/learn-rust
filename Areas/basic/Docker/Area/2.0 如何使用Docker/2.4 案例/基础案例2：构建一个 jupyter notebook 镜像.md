## 使用 Dockerfile 构建一个 jupyter notebook 镜像
[6-2 示例jupyter_sample](../../../Archive/ChatGpt/6-2%20示例jupyter_sample.md)
接下来让我们使用 Docker 来构建一个真实可用的镜像，比如 jupyter notebook 镜像。
```shell

docker build -t jupyter-sample jupyter_sample/

```
该镜像使用 RUN 指令来安装 jupyter notebook，使用 WORKDIR 指令设置工作目录，使用 COPY 指令将代码复制到镜像中，使用 EXPOSE 指令来暴露端口，最后使用 CMD 指令来启动 jupyter notebook 服务。
使用上述镜像来启动 jupyter notebook 服务。

```shell

docker run -d -p 8888:8888  jupyter-sample

```

## 使用多阶段构建来打包一个 golang 应用

  ![](../../asserts/Pasted%20image%2020250724183547.png)

在实际开发中，我们经常需要构建 golang 应用。

如果使用传统的单阶段构建，最终的镜像会包含整个 Go 开发环境，导致镜像体积非常大。

通过多阶段构建，我们可以创建一个非常小的生产镜像。

  

创建一个 [main.go](./golang_sample/main.go) 文件，

一个普通构建的 [Dockerfile](./golang_sample/Dockerfile.single)

以及一个多阶段构建的 [Dockerfile](./golang_sample/Dockerfile.multi)

  

构建镜像：

  

```shell

docker build -t golang-demo-single -f golang_sample/Dockerfile.single golang_sample/

docker build -t golang-demo-multe -f golang_sample/Dockerfile.multi golang_sample/

```

  

运行容器：

  

```shell

docker run -d -p 8080:8080 golang-demo-single

docker run -d -p 8081:8081 golang-demo-multe

```

  

容器运行成功后可以通过如下命令行来访问，可以看到两个容器都是在运行我们写的 golang 服务。

  

```shell

curl http://localhost:8080

curl http://localhost:8081

```

  

让我们来对比一下单阶段构建和多阶段构建的区别：

  

```shell

# 查看镜像大小

docker images | grep golang-demo

```

  

你会发现最终的镜像只有几十 MB，而如果使用单阶段构建（直接使用 golang 镜像），镜像大小会超过 1GB。这就是多阶段构建的优势：

  

- 最终镜像只包含运行时必需的文件

- 不包含源代码和构建工具，提高了安全性

- 大大减小了镜像体积，节省存储空间和网络带宽

  

	这种构建方式特别适合 Go 应用，因为 Go 可以编译成单一的静态二进制文件。在实际开发中，我们可以使用这种方式来构建和部署高效的容器化 Go 应用。