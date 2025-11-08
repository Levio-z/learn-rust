此服务从 `$GIT_URL` 指向的存储库中读取。

客户端必须首先使用 _$GIT_URL/info/refs？service=git-receive-pack_ 中。

```
C: POST $GIT_URL/git-receive-pack HTTP/1.0
C: Content-Type: application/x-git-receive-pack-request
C:
C: ....0a53e9ddeaddad63ad106860237bbf53411d11a7 441b40d833fdfa93eb2908e52742248faf0ee993 refs/heads/maint\0 report-status
C: 0000
C: PACK....

S: 200 OK
S: Content-Type: application/x-git-receive-pack-result
S: Cache-Control: no-cache
S:
S: ....
```

客户端不得重用或重新验证缓存的响应。服务器必须包含足够的 Cache-Control 标头，以防止缓存响应。

服务器应支持此处定义的所有功能。

客户端必须在请求正文中发送至少一个命令。在请求正文的命令部分中，客户端应将通过 ref 发现获得的 id 作为 old_id 发送。

```
update_request  =  command_list
     "PACK" <binary data>

command_list    =  PKT-LINE(command NUL cap_list LF)
     *(command_pkt)
command_pkt     =  PKT-LINE(command LF)
cap_list        =  *(SP capability) SP

command         =  create / delete / update
create          =  zero-id SP new_id SP name
delete          =  old_id SP zero-id SP name
update          =  old_id SP new_id SP name
```

TODO：进一步记录这一点。
