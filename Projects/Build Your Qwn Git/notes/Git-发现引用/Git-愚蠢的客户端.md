仅支持“哑”协议的 HTTP 客户端必须通过请求存储库的特殊 info/refs 文件来发现引用。

哑 HTTP 客户端必须向 `$GIT_URL/info/refs` 发出 `GET` 请求，没有任何搜索/查询参数。

```
C: GET $GIT_URL/info/refs HTTP/1.0

S: 200 OK
S:
S: 95dcfa3633004da0049d3d0fa03f80589cbcaf31	refs/heads/maint
S: d049f6c27a2244e12041955e262a404c7faba355	refs/heads/master
S: 2cb58b79488a98d2721cea644875a8dd0026b115	refs/tags/v1.0
S: a3c2e2402b99163d1d59756e5f207ae21cccba4c	refs/tags/v1.0^{}
```

返回的 info/refs 实体的 Content-Type 应该是 _text/plain; 字符集=utf-8_，但可以是任何内容类型。 客户端不得尝试验证返回的 Content-Type。 哑服务器不得返回以 `application/x-git-` 的

可以返回 Cache-Control 标头以禁用返回实体的缓存。

检查响应时，客户端应该只检查 HTTP 状态代码。有效响应为 `200` `OK` 或 `304` `Not` `Modified`。

返回的内容是描述每个引用及其已知值的 UNIX 格式的文本文件。文件应根据 C 语言环境顺序按名称排序。该文件不应包含名为 `HEAD` 的默认引用。

```
info_refs   =  *( ref_record )
ref_record  =  any_ref / peeled_ref

any_ref     =  obj-id HTAB refname LF
peeled_ref  =  obj-id HTAB refname LF
 obj-id HTAB refname "^{}" LF
```

