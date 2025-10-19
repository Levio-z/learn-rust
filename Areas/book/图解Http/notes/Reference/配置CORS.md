
配置 CORS 实际上是让 Spring Boot 在其 HTTP 响应中添加必要的 **CORS 头部字段**，例如：

| 头部字段                           | 目的                                                     |
| ------------------------------ | ------------------------------------------------------ |
| `Access-Control-Allow-Origin`  | 告诉浏览器，允许哪个前端源访问我的数据。                                   |
| `Access-Control-Allow-Methods` | 告诉浏览器，允许哪些 HTTP 方法（GET, POST, PUT, DELETE）跨域访问。        |
| `Access-Control-Allow-Headers` | 告诉浏览器，允许前端请求携带哪些自定义的 HTTP 头部（如 `Authorization` Token）。 |