支持“智能”协议（或“智能”和“哑”协议）的 HTTP 客户端必须通过对存储库的 info/refs 文件发出参数化请求来发现引用。

请求必须只包含一个查询参数， `service=$servicename`，其中 `$servicename` 必须是客户端希望联系以完成作的服务名称。请求不得包含其他查询参数。


```
C: GET $GIT_URL/info/refs?service=git-upload-pack HTTP/1.0
```

dumb server reply:  哑服务器回复：
```
S: 200 OK
S:
S: 95dcfa3633004da0049d3d0fa03f80589cbcaf31	refs/heads/maint
S: d049f6c27a2244e12041955e262a404c7faba355	refs/heads/master
S: 2cb58b79488a98d2721cea644875a8dd0026b115	refs/tags/v1.0
S: a3c2e2402b99163d1d59756e5f207ae21cccba4c	refs/tags/v1.0^{}
```

smart server reply:  智能服务器回复：
```
S: 200 OK
S: Content-Type: application/x-git-upload-pack-advertisement
S: Cache-Control: no-cache
S:
S: 001e# service=git-upload-pack\n
S: 0000
S: 004895dcfa3633004da0049d3d0fa03f80589cbcaf31 refs/heads/maint\0multi_ack\n
S: 003fd049f6c27a2244e12041955e262a404c7faba355 refs/heads/master\n
S: 003c2cb58b79488a98d2721cea644875a8dd0026b115 refs/tags/v1.0\n
S: 003fa3c2e2402b99163d1d59756e5f207ae21cccba4c refs/tags/v1.0^{}\n
S: 0000
```

客户端可以在 Git-Protocol HTTP 标头中将额外参数（参见 文档/技术/pack-protocol.txt）作为冒号分隔的字符串发送。

使用 `--http-backend-info-refs` 选项来 [git-upload-pack[1]](https://git-scm.com/docs/git-upload-pack) 中。