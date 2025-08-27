```
# 练习 3：Rust 多阶段构建

#

# 要求：

# 1. 使用多阶段构建来优化 Rust 应用的 Docker 镜像

# 2. 第一阶段：

#    - 使用 rust:1.75-slim 作为基础镜像

#    - 设置工作目录

#    - 复制 Cargo.toml 和 Cargo.lock（如果存在）

#    - 复制源代码

#    - 安装 MUSL 目标环境, 支持交叉编译 rustup target add x86_64-unknown-linux-musl

#    - 使用 cargo build --target x86_64-unknown-linux-musl --release 构建应用

# 3. 第二阶段：

#    - 使用 alpine:latest 作为基础镜像

#    - 从第一阶段复制编译好的二进制文件

#    - 设置工作目录

#    - 运行应用

#

# 提示：

# 1. 使用 COPY --from=builder 从构建阶段复制文件

# 2. 注意文件权限和所有权

# 3. 确保最终镜像尽可能小, 小于 20 M

#

# 测试命令：

# docker build -t rust-exercise3 .

# docker run rust-exercise3

  

# 在这里编写你的 Dockerfile

  

# 第一阶段：构建阶段

FROM rust:1.75-slim AS builder

  

# 安装交叉编译环境

RUN rustup target add x86_64-unknown-linux-musl

  

# 设置工作目录

WORKDIR /app

# 复制 Cargo.toml 和 Cargo.lock（如果存在）

COPY Cargo.toml  ./

# 将源代码复制到容器中

COPY src/ ./src/

  

# 构建应用

RUN cargo build --target x86_64-unknown-linux-musl --release

  

# 第二阶段：运行阶段

FROM alpine:latest

  

# 设置工作目录

WORKDIR /app

COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rust-docker-example .

  

# 如果需要执行权限（通常已具备）

RUN chmod +x ./rust-docker-example

# 运行应用

CMD ["./rust-docker-example"]
```
### cargo.toml优化
```
[package]
name = "rust-docker-example"
version = "0.1.0"
edition = "2021"

[dependencies]
chrono = "0.4" 

[profile.release]
opt-level = "z"         # 为最小体积优化（不是最大速度）18.3
lto = true              # 开启链接时间优化（Link Time Optimization）15.5
strip = "symbols"       # 1.75+ 可内置 strip，无需单独 strip 命令  13.2
# codegen-units = 1       # 减少并行代码生成单元，有助 LTO 整体优化 13.2

```