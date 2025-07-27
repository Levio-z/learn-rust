```
# 基础镜像

FROM python:3.11-slim

# 定义构建参数

ARG PORT=5000

# 设置环境变量，避免 Python 缓存文件生成
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
    
# 设置工作目录

WORKDIR /app

  

# 安装系统依赖（变更频率低）

RUN apt-get update && \

    apt-get install -y --no-install-recommends  && \

    rm -rf /var/lib/apt/lists/*

  

# 安装应用依赖（变更频率中等）

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

  

# 复制应用代码（变更频率高）

COPY app.py .

  

# 创建非 root 用户并设置权限

RUN useradd -s /bin/sh appuser 

USER appuser

  

# 环境变量和端口

ENV PORT=${PORT}

EXPOSE ${PORT}

  

# 启动应用

CMD ["python", "app.py"]
```