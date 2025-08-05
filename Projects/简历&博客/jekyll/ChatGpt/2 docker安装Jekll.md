	### 创建一个my-jekyll-site
```
my-jekyll-site/
├── _config.yml          # 最简单的 Jekyll 配置
├── Gemfile              # 依赖管理
└── docker-compose.yml   # Docker Compose 配置
└── index.md   # 首页
```
- `_config.yml`（示例内容）：
```
title: Demo Jekyll Site
baseurl: ""
```
- index.md
```
---
title: 首页
---

欢迎访问我的 Jekyll 网站！
```
 `Gemfile`
```
source "https://rubygems.org"

gem "jekyll", "~> 4.3"
gem "webrick", "~> 1.7"   # Ruby 3.0+ 需要显式添加 Webrick
```
完整的 `docker-compose.yml`
```
version: '3.8'

services:
  jekyll:
    image: jekyll/jekyll:4.3
    command: jekyll serve --watch --force_polling --host 0.0.0.0
    volumes:
      - ./:/srv/jekyll
    ports:
      - "4000:4000"
    environment:
      - JEKYLL_ENV=development
    stdin_open: true
    tty: true
```
### 说明

- `image`: 官方 Jekyll 镜像（这里固定版本为 4.3）
    
- `command`: 启动 Jekyll 服务器，`--host 0.0.0.0` 使容器内服务对外可见，`--force_polling` 在某些共享卷环境（如 Windows、Docker Desktop）下保证文件变化被检测到
    
- `volumes`: 把当前目录挂载到容器的默认 Jekyll 工作目录 `/srv/jekyll`
    
- `ports`: 容器的 4000 端口映射到宿主机 4000 端口
    
- `stdin_open` 和 `tty`：让容器支持交互

### 如何启动
初始依赖(执行一次)
```
docker-compose run --rm jekyll bundle install
```
启动服务：
```
docker-compose up -d
```
浏览器访问:
```
http://localhost:4000
```
