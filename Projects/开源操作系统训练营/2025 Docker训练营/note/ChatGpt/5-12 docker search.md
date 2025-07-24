```
docker search  mysql
```

### help

```
➜  /workspace git:(main) docker search --help           

Usage:  docker search [OPTIONS] TERM

Search Docker Hub for images

Options:
  -f, --filter filter   Filter output based on conditions provided
      --format string   Pretty-print search using a Go template
      --limit int       Max number of search results
      --no-trunc        Don't truncate output
```
## 常用选项详解

| 选项                          | 含义                      |
| --------------------------- | ----------------------- |
| `--filter`, `-f`            | 使用条件过滤结果，如 `stars=100`  |
| `--limit`                   | 限制结果数量（默认最多返回 25 个）     |
| `--no-trunc`                | 不截断输出内容（完整显示描述）         |
| `--format`                  | 使用 Go 模板格式化输出           |
| `--platform`（Docker 20.10+） | 指定平台架构（如 `linux/amd64`） |
### 应用示例
```
docker search nginx --filter=stars=500
	``` 