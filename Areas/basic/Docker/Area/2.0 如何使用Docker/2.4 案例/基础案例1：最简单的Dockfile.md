### Dockefile文件构建详解
```
# 阶段一
# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 避免交互式安装提示（设置非交互模式）
ENV DEBIAN_FRONTEND=noninteractive

# 定义构建参数
ARG PORT=80
# 阶段二
# 更新包索引并安装 nginx 和 curl
RUN apt-get update && \
    apt-get install -y nginx curl && \
    rm -rf /var/lib/apt/lists/*
# 阶段三 无应用依赖，省略
# 阶段四 应用 create、copy文件
# 创建 HTML 文件，内容为 "Hello Docker!"
RUN echo "<!DOCTYPE html><html><head><title>Welcome</title></head><body><h1>Hello Docker!</h1></body></html>" \
    > /var/www/html/index.html
# 阶段五 暴露端口和配置环境变量
# 暴露 nginx 默认端口 80
EXPOSE ${PORT}

# 设置环境变量
ENV PORT=${PORT}
# 阶段六
# 启动 nginx 服务（使用前台模式，以防容器退出）
CMD ["nginx", "-g", "daemon off;"]
```
