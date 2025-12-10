此服务从 `$GIT_URL` 指向的存储库中读取。

客户端必须首先使用 _$GIT_URL/info/refs？service=git-upload-pack_ 中。

```
C: POST $GIT_URL/git-upload-pack HTTP/1.0
C: Content-Type: application/x-git-upload-pack-request
C:
C: 0032want 0a53e9ddeaddad63ad106860237bbf53411d11a7\n
C: 0032have 441b40d833fdfa93eb2908e52742248faf0ee993\n
C: 0000

S: 200 OK
S: Content-Type: application/x-git-upload-pack-result
S: Cache-Control: no-cache
S:
S: ....ACK %s, continue
S: ....NAK
```
客户端不得重用或重新验证缓存的响应。服务器必须包含足够的 Cache-Control 标头，以防止缓存响应。


服务器应支持此处定义的所有功能。

客户端必须在请求正文中发送至少一个“want”命令。客户端不得引用“want”命令中的 id，该 id 未出现在通过 ref 发现获得的响应中，除非服务器通告功能`allow-tip-sha1-in-want`或 `allow-reachable-sha1-in-want`。



```
compute_request   =  want_list
       have_list
       request_end
request_end       =  "0000" / "done"

want_list         =  PKT-LINE(want SP cap_list LF)
       *(want_pkt)
want_pkt          =  PKT-LINE(want LF)
want              =  "want" SP id
cap_list          =  capability *(SP capability)

have_list         =  *PKT-LINE("have" SP id LF)
```

