 在此阶段，您将发送带有相关 ID 的响应。

### Response message  响应消息

Kafka brokers communicate with clients through the [Kafka wire protocol](https://kafka.apache.org/protocol.html). The protocol uses a request-response model, where the client sends a request message and the broker replies with a response message.  
Kafka 代理通过 [Kafka 有线协议](https://kafka.apache.org/protocol.html)与客户端通信。该协议使用请求-响应模型，其中客户端发送请求消息，代理回复响应消息。

A Kafka response message has three parts:  
Kafka 响应消息由三个部分组成：

1. `message_size`
2. Header  页眉
3. Body  身体

For this stage, you can ignore the body and just focus on `message_size` and the header. You'll learn about response bodies in a later stage.  
对于此阶段，您可以忽略正文，只关注 `message_size` 和标题。您将在稍后的阶段了解响应正文。

#### The `message_size` field  
`message_size` 字段

The [`message_size`](https://kafka.apache.org/protocol.html#protocol_common) field is a 32-bit signed integer. It specifies the size of the header and body.  
[`message_size`](https://kafka.apache.org/protocol.html#protocol_common) 字段是 32 位有符号整数。它指定标题和正文的大小。

For this stage, the tester will only assert that your `message_size` field is 4 bytes long—it won't check the value. You'll implement correct `message_size` values in a later stage.  
对于此阶段，测试人员只会断言您的 `message_size` 字段长度为 4 字节，它不会检查该值。您将在稍后的阶段实现正确的 `message_size` 值。

#### Header  页眉

Kafka has a few different header versions. The way Kafka determines which header version to use is a bit complicated and is outside the scope of this challenge. For more information, take a look at [KIP-482](https://cwiki.apache.org/confluence/display/KAFKA/KIP-482%3A+The+Kafka+Protocol+should+Support+Optional+Tagged+Fields) and this [Stack Overflow answer](https://stackoverflow.com/a/71853003).  
Kafka 有几个不同的标头版本。Kafka 确定使用哪个标头版本的方式有点复杂，并且超出了本次挑战的范围。有关更多信息，请查看 [KIP-482](https://cwiki.apache.org/confluence/display/KAFKA/KIP-482%3A+The+Kafka+Protocol+should+Support+Optional+Tagged+Fields) 和此 [Stack Overflow 答案](https://stackoverflow.com/a/71853003) 。

In this stage, you will use [response header v0](https://kafka.apache.org/protocol.html#protocol_messages) (scroll down).  
在此阶段，您将使用[响应标头 v0](https://kafka.apache.org/protocol.html#protocol_messages)（向下滚动）。

Response header v0 contains a single field: [`correlation_id`](https://developer.confluent.io/patterns/event/correlation-identifier/). This field lets clients match responses to their original requests. Here's how it works:  
响应标头 v0 包含一个字段：[`correlation_id`](https://developer.confluent.io/patterns/event/correlation-identifier/)。此字段允许客户端将响应与其原始请求进行匹配。它的工作原理如下：

1. The client generates a correlation ID.  
    客户端生成一个相关 ID。
2. The client sends a request that includes the correlation ID.  
    客户端发送包含相关 ID 的请求。
3. The broker sends a response that includes the same correlation ID.  
    代理发送包含相同相关 ID 的响应。
4. The client receives the response and matches the correlation ID to the original request.  
    客户端接收响应并将关联 ID 与原始请求匹配。

The `correlation_id` field is a 32-bit signed integer. For this stage, your program must respond with a hard-coded `correlation_id` of 7.  
`correlation_id` 字段是 32 位有符号整数。对于此阶段，程序必须以硬编码 `correlation_id` 7 进行响应。