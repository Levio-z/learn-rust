ollama/Dockfile
```
FROM cnbcool/default-dev-env:latest

  

ARG DEBIAN_FRONTEND=noninteractive

  

RUN apt update

  

RUN curl -fsSL https://ollama.com/install.sh | sh

  

ENV OLLAMA_MODELS=./deepseek-models

  

ENV OLLAMA_HOST=0.0.0.0

  

ENV OLLAMA_ORIGINS=*

  

# 设置环境变量（会在运行时生效）

ENV OLLAMA_FLASH_ATTENTION=1

  

# 容器启动时运行服务

CMD ["ollama", "serve"]
```

### .env
```
# RAGWEBUI 配置

CHAT_PROVIDER=ollama

OLLAMA_API_BASE=http://host.docker.internal:11434

OLLAMA_MODEL=deepseek-r1:8b-0528-qwen3-q8_0

EMBEDDINGS_PROVIDER=ollama

OLLAMA_EMBEDDINGS_MODEL=nomic-embed-text

  

# 向量数据库配置

VECTOR_STORE_TYPE=chroma

CHROMA_DB_HOST=chromadb

CHROMA_DB_PORT=8000

  

# MySQL 配置

MYSQL_SERVER=db

MYSQL_USER=ragwebui

MYSQL_PASSWORD=ragwebui

MYSQL_DATABASE=ragwebui

  

# MinIO 配置

MINIO_ENDPOINT=minio:9000

MINIO_ACCESS_KEY=minioadmin

MINIO_SECRET_KEY=minioadmin

MINIO_BUCKET_NAME=documents
```

### cnb
```
        image: docker.cnb.cool/ai-models/deepseek-ai/deepseek-r1-0528-qwen3-8b-run-with-ollama/dockerfile-caches:e190b668e7ee96359c39ef34f9639e030d302bba
```

### compose
```
services:

  backend:

    build: ./backend

    env_file:

      - .env

    volumes:

      - ./backend:/app

      - ./uploads:/app/uploads

    networks:

      - app_network

    depends_on:

      db:

        condition: service_healthy

      chromadb:

        condition: service_started

      minio:

        condition: service_started

    restart: on-failure

    deploy:

      restart_policy:

        condition: on-failure

        delay: 5s

        max_attempts: 3

  

  frontend:

    build: ./frontend

    volumes:

      - ./frontend:/app

      - /app/node_modules

    networks:

      - app_network

  

  db:

    image: mysql:8.0

    command: --default-authentication-plugin=mysql_native_password

    environment:

      - MYSQL_ROOT_PASSWORD=root

      - MYSQL_DATABASE=ragwebui

      - MYSQL_USER=ragwebui

      - MYSQL_PASSWORD=ragwebui

      - TZ=Asia/Shanghai

    ports:

      - "3306:3306"

    volumes:

      - mysql_data:/var/lib/mysql

    networks:

      - app_network

    healthcheck:

      test:

        [

          "CMD",

          "mysqladmin",

          "ping",

          "-h",

          "localhost",

          "-u",

          "$$MYSQL_USER",

          "--password=$$MYSQL_PASSWORD",

        ]

      interval: 5s

      timeout: 5s

      retries: 5

      start_period: 10s

  

  chromadb:

    image: chromadb/chroma:latest

    ports:

      - "8001:8000"

    volumes:

      - chroma_data:/chroma/chroma

    networks:

      - app_network

  

  # For Qdrant, Remove the comment and run the following command to start the service

  # qdrant:

  #   image: qdrant/qdrant:latest

  #   ports:

  #     - "6333:6333" # REST API

  #     - "6334:6334" # GRPC

  #   volumes:

  #     - qdrant_data:/qdrant/storage

  #   environment:

  #     - QDRANT_ALLOW_RECOVERY_MODE=true

  #   networks:

  #     - app_network

  

  minio:

    image: minio/minio:latest

    ports:

      - "9000:9000" # API port

      - "9001:9001" # Console port

    environment:

      - MINIO_ROOT_USER=minioadmin

      - MINIO_ROOT_PASSWORD=minioadmin

    volumes:

      - minio_data:/data

    command: server --console-address ":9001" /data

    networks:

      - app_network

  

  nginx:

    image: nginx:alpine

    ports:

      - "80:80"

    volumes:

      - ./nginx.conf:/etc/nginx/nginx.conf:ro

    depends_on:

      - frontend

      - backend

      - minio

    networks:

      - app_network

  

  ollama:

    image: docker.cnb.cool/ai-models/deepseek-ai/deepseek-r1-0528-qwen3-8b-run-with-ollama/dockerfile-caches:e190b668e7ee96359c39ef34f9639e030d302bba

    ports:

      - "11434:11434"  # 把容器的 Ollama API 暴露给宿主机

    environment:

      - OLLAMA_MODELS=/workspace/models

      - OLLAMA_ORIGINS=*

      - OLLAMA_HOST=0.0.0.0  # 启动时监听所有接口，便于外部访问

      - OLLAMA_FLASH_ATTENTION=1

    volumes:

      - ./deepseek-models:/workspace/models  # 持久化模型存储目录

    restart: unless-stopped

    command: ollama serve

    networks:

      - app_network

volumes:

  mysql_data:

  chroma_data:

  minio_data:

  # qdrant_data:

  

networks:

  app_network:

    driver: bridge
```

### nginx
```
            proxy_redirect http:// https://;
```