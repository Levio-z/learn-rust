 将缓冲区写入此写入器，返回写入的字节数。

 此函数将尝试写入 `buf` 的整个内容，但整个写入可能不会成功，或者写入也可能生成错误。通常，对 `write` 的调用表示对任何包装对象进行写入的一次尝试。
 对 `write 的`调用不能保证在等待数据写入时阻塞，否则会阻塞的写入可以通过 [`Err`](https://doc.rust-lang.org/std/result/enum.Result.html#variant.Err "variant std::result::Result::Err") 变量来指示。
 如果此方法消耗了 `n 个> 0` 字节的 `buf`，则必须返回 [`Ok（n）`](https://doc.rust-lang.org/std/result/enum.Result.html#variant.Ok "variant std::result::Result::Ok")。如果返回值为 `Ok（n）`，则 `n` 必须满足 `n <= buf.len（）`。返回值 `Ok（0）` 通常意味着基础对象不再能够接受字节，并且将来也可能无法接受，或者提供的缓冲区为空。
 **Errors  错误**
 - 每次调用`写`操作都可能生成一个 I/O 错误，指示操作无法完成。如果返回错误，则缓冲区中没有字节写入此写入器。
 - 如果无法将整个缓冲区写入此写入器，则**不会将**其视为错误。