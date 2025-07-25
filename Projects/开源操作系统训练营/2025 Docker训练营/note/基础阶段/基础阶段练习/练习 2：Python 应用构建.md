```
# 练习 2：Python 应用构建

#

# 要求：

# 1. 创建一个简单的 Python Flask 应用的 Dockerfile

# 2. 实现以下功能：

#    - 使用 python:3.11-slim 作为基础镜像

#    - 安装应用依赖

#    - 创建非 root 用户运行应用

#    - 配置工作目录

#    - 设置环境变量

#    - 暴露应用端口

#

# 提示：

# - 使用 requirements.txt 管理依赖

# - 使用 WORKDIR 设置工作目录

# - 使用 USER 指令切换用户

# - 使用 ENV 设置环境变量

#

# 测试命令：

# docker build -t exercise2 .

# docker run -d -p 5000:5000 exercise2

# curl -s http://127.0.0.1:8080 | grep -q "Hello Docker" && echo "Found" || echo "Not Found"

  

# 在这里编写你的 Dockerfile 指令

  
  
  

FROM python:3.11-slim

  

# 创建用户 appuser，不创建 home 目录，shell 设置为 /bin/bash（或 /bin/sh）

RUN useradd -m -s /bin/bash appuser

  

# 切换用户

USER appuser

  

# 定义构建参数

ARG PORT=5000

# 设置环境变量

ENV PORT=${PORT}  

  

WORKDIR /app

COPY app.py requirements.txt ./

# 使用 Python 的包管理工具 pip，安装 requirements.txt 文件中列出的所有依赖包。

RUN pip install -r requirements.txt

  

EXPOSE ${PORT}

  

CMD ["python", "app.py"]
```