
在构建现代的 **RESTful API**（例如为移动应用或前端框架提供数据服务）时，通常会使用一套更完整的 HTTP 方法来对应资源的 CRUD 操作（Create, Read, Update, Delete），这些方法包括：

| HTTP 方法    | 对应 RESTful API 操作     | 传统网站中（非 API 场景）        |
| ---------- | --------------------- | ---------------------- |
| **GET**    | Read (读取)             | 广泛使用 (加载页面、内容)         |
| **POST**   | Create (创建)           | 广泛使用 (提交表单、登录)         |
| **PUT**    | Update/Replace (完全替换) | **极少使用** (因安全顾虑)       |
| **DELETE** | Delete (删除)           | **极少使用** (通常用 POST 模拟) |
| **PATCH**  | Update/Modify (部分修改)  | **极少使用**               |
