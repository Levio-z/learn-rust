## USER 指令作用

-   切换为指定用户名或 UID，后续所有命令都以该用户身份执行。
    
-   主要用于限制权限，提升容器安全。
    
-   常用于切换到非 root 用户运行应用。
    

---

## 基本用法

```dockerfile
USER <username>[:<groupname>]
```

-   `<username>`：用户名或 UID。
    
-   `<groupname>`：可选，用户组名或 GID。
    

---

## 示例

假设 Dockerfile 中：

```dockerfile
FROM ubuntu:22.04

# 创建新用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 切换用户
USER appuser

# 下面命令都以 appuser 用户执行
RUN whoami  # 输出 appuser
```

---

## 使用场景

1.  **创建非 root 用户**
    

```dockerfile
RUN groupadd -r mygroup && useradd -r -g mygroup myuser
```

2.  **切换用户**
    

```dockerfile
USER myuser
```

3.  **运行应用**
    

```dockerfile
CMD ["./start-app.sh"]
```

---

## 注意事项

-   如果切换的用户不存在，构建时会报错。
    
-   `USER` 指令影响后续所有命令，包括 `RUN`、`CMD`、`ENTRYPOINT`。
    
-   通常建议：
    
    -   安装、配置时用 root（默认）。
        
    -   启动应用时切换到非 root 用户。
        

---

## 结合 Dockerfile 简单示例

```dockerfile
FROM python:3.11-slim

# 创建用户
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 复制代码
COPY app.py /app/

# 切换用户
USER appuser

WORKDIR /app

CMD ["python", "app.py"]
```

---