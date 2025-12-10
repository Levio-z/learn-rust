如果服务器无法识别请求的服务名称，或者服务器管理员已禁用请求的服务名称，则服务器必须使用 `403` `Forbidden` HTTP 状态代码进行响应。

否则，智能服务器必须使用请求的服务名称的智能服务器回复格式进行响应。

Cache-Control 标头“应”用于禁用返回实体的缓存。

Content-Type 必须是 `application/x-$servicename-advertisement` 。如果返回另一种内容类型，客户端“应该”回退到哑协议。当回退到哑协议时，客户端“不应该”向 `$GIT_URL/info/refs` 发出额外的请求，而应该使用手头已有的响应。如果客户端不支持哑协议，则不得继续。


客户端必须验证状态代码是否为 `200` `OK` 或 `304` `未``修改` 。


客户端必须验证响应实体的前五个字节是否与正则表达式 `^`[`0-9a-f`]`{4}#` 匹配。如果此测试失败，客户端不得继续。


客户端必须将整个响应解析为一系列 pkt-line 记录。


客户端必须验证第一个 pkt-line 是 `#` `service=$servicename`。服务器必须将$servicename 设置为请求参数值。服务器应在此行末尾包含一个 LF。客户端必须忽略行尾的 LF。


服务器必须使用神奇的 `0000` end pkt-line 标记终止响应。

返回的响应是描述每个引用及其已知值的 pkt-line 流。流应根据 C 语言环境顺序按名称排序。流应该包含名为 `HEAD` 的默认引用作为第一个引用。流必须在第一个引用的 NUL 后面包含功能声明。

如果“version=1”作为额外参数发送，则返回的响应包含“version 1”。

```
smart_reply     =  PKT-LINE("# service=$servicename" LF)
     "0000"
     *1("version 1")
     ref_list
     "0000"
ref_list        =  empty_list / non_empty_list

empty_list      =  PKT-LINE(zero-id SP "capabilities^{}" NUL cap-list LF)

non_empty_list  =  PKT-LINE(obj-id SP name NUL cap_list LF)
     *ref_record

cap-list        =  capability *(SP capability)
capability      =  1*(LC_ALPHA / DIGIT / "-" / "_")
LC_ALPHA        =  %x61-7A

ref_record      =  any_ref / peeled_ref
any_ref         =  PKT-LINE(obj-id SP name LF)
peeled_ref      =  PKT-LINE(obj-id SP name LF)
     PKT-LINE(obj-id SP name "^{}" LF
```


