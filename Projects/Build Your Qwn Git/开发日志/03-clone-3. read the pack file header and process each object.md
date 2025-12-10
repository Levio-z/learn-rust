### 1 参考
#### 1.1 挑战内容
- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/discovering-references
- 使用智能协议请求仓库将内容下载下来
#### 1.2 资料
- [This forum post](https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/intro)
	- 如何逐步实现这一点的建议
- Git 的[智能 HTTP 传输协议](https://www.git-scm.com/docs/http-protocol)
- [Git-gitprotocol-pack - 0-如何通过网络传输软件包](../notes/Git-智能协议/Git-gitprotocol-pack%20-%200-如何通过网络传输软件包.md)
	- [Git-gitprotocol-pack - 3-如何通过网络传输软件包-packfile文件格式](../notes/Git-智能协议/Git-gitprotocol-pack%20-%203-如何通过网络传输软件包-packfile文件格式.md)
#### 1.3 知识点
- 文件相关：[rust-std-fs-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/fs/rust-std-fs-TOC.md)
- zlibDecoder：[ZlibEncoder-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/flate2/struct/ZlibEncoder/ZlibEncoder-TOC.md)
- 什么是c风格字符串：[Rust  CStr TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/ffi/Cstr/Rust%20%20CStr%20TOC.md)
- 分割字符串：[Rust `&str` 常用字符串操作方法概览](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/core/str/常用方法/Rust%20`&str`%20常用字符串操作方法概览.md)
- 装饰器（限制读取）：[take](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/BufRead/take.md)
- 解析为数值：[解析为数值](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/core/str/常用方法/解析为数值.md)
- print的底层：[rust-stdout-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/stdout/rust-stdout-TOC.md)
	- [copy reader to writer](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/methods/io%20copy/copy%20reader%20to%20writer.md)
	- [writeln! 与write_all](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/stdout/研究/writeln!%20与write_all.md)
- 错误处理： [rust-anyhow-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/anyhow/rust-anyhow-TOC.md)
- 临时文件：[rust-crate-tempfile-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/tempfile/rust-crate-tempfile-TOC.md)
- 命令行工具：[rust-clap-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/clap/rust-clap-TOC.md)
- 文件格式和基本流程梳理(建议单独开启一个网址来参考)：[Git Object对象](../../../Areas/basic/Git/Pro%20Git/8.0%20Git底层原理/内存模型/对象/Git%20Object对象.md)
- 闭包：[rust-闭包-TOC](../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.4%20函数式编程特性/2.4.1%20闭包/rust-闭包-TOC.md)
- 迭代器：[rust-iter-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/iter/rust-iter-TOC.md)
- 字符串去除尾部：[Rust-String-去除结尾特定内容的方法-toc](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/alloc/String/methods/Rust-String-去除结尾特定内容的方法-toc.md)
- reqwest：[Rust-reqwest-基本概念-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/reqwest/Rust-reqwest-基本概念-TOC.md)
	- [Rust-Bytes-基本概念](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/crate/reqwest/Rust-Bytes-基本概念.md)
- 切片和数组的git方法：
	- [Rust-slice 切片-get(..)](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-get(..).md)
	- [Rust-slice 切片-split_first_chunk-固定长度切割切片](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/slice/methods/Rust-slice%20切片-split_first_chunk-固定长度切割切片.md)
- 链式处理：[Rust-Result-and_then](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/option/Option/methods/Rust-Result-and_then.md)
- Cursor：[Rust-cursor-基本概念-TOC](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/Cursor/Rust-cursor-基本概念-TOC.md)
- 缓冲读：[Rust-io-BufReader](../../../Areas/Rust/Area/1%20基本概念/3%20库/库/标准库/std/io/BufRead/Rust-io-BufReader.md)
### 2 方案设计
#### 2.1 解析协商的一部分”0008NAK\n“
```rust
    let mut nak_vec = vec![0; 8];

    bufreader

        .read_exact(&mut nak_vec)

        .context("read nak fail")?;
	if nak_vec != b"0008NAK\n" {

        anyhow::bail!("expected NAK, got {:?}", String::from_utf8_lossy(&nak_vec));

    }
```
### 2.2 解析packfile 头部

```rust
let mut bufreader = HashReader::new(bufreader);

    let mut pack = vec![0; 4];

    bufreader.read_exact(&mut pack).context("read pack fail")?;

    if pack != b"PACK" {

        anyhow::bail!("expected PACK, got {:?}", String::from_utf8_lossy(&pack));

    }

  

    // 版本号，接下来四字节

    let packfile_version = bufreader

        .read_u32::<BigEndian>()

        .context("read packfile version fail")?;

    eprintln!("packfile_version: {}", packfile_version);

  

    // 打包文件数量，接下来四字节

    let num_objects = bufreader

        .read_u32::<BigEndian>()

        .context("read packfile object count fail")?;

    eprintln!("num_objects: {}", num_objects);
```

### 2.3 解析出Object
#### 2.3.0 方法
##### 获取对象大小
```rust
fn read_size<R: BufRead>(reader: &mut R, is_delta: bool) -> anyhow::Result<(usize, u8)> {

    let mut byte = reader.read_u8().context("read object type fail")?;

    let obj_type = (byte >> 4) & 0b111;

    let mut size = if is_delta {

        byte & 0b0111_1111

    } else {

        byte & 0b0000_1111

    } as usize;

    let mut shift = if is_delta { 7 } else { 4 };

    // 继续读取后续字节，直到最高位为 0

    while (byte & 0b1000_0000) != 0 {

        byte = reader.read_u8().context("read object size fail")?;

        size |= ((byte & 0b0111_1111) as usize) << shift;

        shift += 7;

    }

    Ok((size, obj_type))

}
```
[Git-Parsing packfiles  object entry 格式](../notes/deltification%20概念/Git-Parsing%20packfiles%20%20object%20entry%20格式.md)
#### 读取一个完整的压缩数据
```rust
fn read_one_object<R: BufRead>(reader: &mut R) -> anyhow::Result<Cursor<Vec<u8>>> {

    // 不需要 BufReader 包 ZlibDecoder，直接解码即可

    let mut decoder = ZlibDecoder::new(reader);

    let mut buf = Vec::new();

    decoder.read_to_end(&mut buf)?;

    Ok(Cursor::new(buf))

}
```
[Rust-ZlibDecoder-使用bufread避免多读数据](../notes/Reference/Rust-ZlibDecoder-使用bufread避免多读数据.md)
#### 2.3.1 解析普通对象
[Git-gitprotocol-pack - 3-如何通过网络d传输软件包-packfile文件格式](../notes/Git-智能协议/Git-gitprotocol-pack%20-%203-如何通过网络传输软件包-packfile文件格式.md)
[Git-Parsing packfiles  object entry 格式](../notes/deltification%20概念/Git-Parsing%20packfiles%20%20object%20entry%20格式.md)
n-byte type and length (3-bit type, (n-1)*7+4-bit length)
compressed data
```rust
let mut hashmap: HashMap<[u8; 20], Object<Cursor<Vec<u8>>>> = std::collections::HashMap::new();
for object_index in 0..num_objects {
/// 获取大小
let (size, obj_type) = read_size(&mut bufreader, false).context("read object size fail")?;
}
if let Some(obj_type) = ObjectType::from_u8(obj_type) {

            match obj_type {

                ObjectType::Commit | ObjectType::Tree | ObjectType::Blob | ObjectType::Tag => {

                    eprintln!("object is of type {:?}", obj_type);

                    let mut object = Object {

                        kind: Kind::from(obj_type),

                        expected_size: size as u64,

                        reader: read_one_object(&mut bufreader)?,

                    };

                    let hash = object.compute_hash(std::io::sink(), false).await?;

                    object.reader.set_position(0);

                    eprintln!(

                        "Object {} is of type {:?} with hash {:?}",

                        object_index,

                        object.kind,

                        hex::encode(hash)

                    );

                    hashmap.insert(hash, object);

                }
                ...
}
```
- 获取普通对象，计算hash并放入map中用于后面对象构造

#### 2.3.2 解析Git-OBJ_REF_DELTA对象
[Git-OBJ_REF_DELTA-基本概念](../notes/Reference/Git-OBJ_REF_DELTA-基本概念.md)
```rust
      // 读取 base object id

                    let mut base_object_id = [0; 20];

                    bufreader

                        .read_exact(&mut base_object_id)

                        .context("read base object id fail")?;

                    eprintln!("base_object_id: {:?}", hex::encode(base_object_id));

  

                    let hex_base_object_id = base_object_id;

                    let base_object = hashmap

                        .get(&hex_base_object_id)

                        .context("can not find base object")?;

  

                    let mut delta_data = read_one_object(&mut bufreader)?;

                    let (src_size, _) =

                        read_size(&mut delta_data, true).context("read src size fail")?;

                    let (tgt_size, _) =

                        read_size(&mut delta_data, true).context("read tgt size fail")?;

                    eprintln!("src_size: {}, tgt_size: {}", src_size, tgt_size);

                    let mut new_tgt = Vec::with_capacity(tgt_size);

                    while delta_data.position() < delta_data.get_ref().len() as u64 {

                        let opcode = delta_data.read_u8().context("read delta opcode fail")?;

                        if (opcode & 0x80) != 0 {

                            // copy from base

                            let mut copy_offset = 0;

                            let mut copy_size = 0;

                            if (opcode & 0b0000_0001) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy offset byte fail")?;

                                copy_offset |= b as usize;

                            }

                            if (opcode & 0b0000_0010) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy offset byte fail")?;

                                copy_offset |= (b as usize) << 8;

                            }

                            if (opcode & 0b0000_0100) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy offset byte fail")?;

                                copy_offset |= (b as usize) << 16;

                            }

                            if (opcode & 0b0000_1000) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy offset byte fail")?;

                                copy_offset |= (b as usize) << 24;

                            }

  

                            if (opcode & 0b0001_0000) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy size byte fail")?;

                                copy_size |= b as usize;

                            }

                            if (opcode & 0b0010_0000) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy size byte fail")?;

                                copy_size |= (b as usize) << 8;

                            }

                            if (opcode & 0b0100_0000) != 0 {

                                let b = delta_data

                                    .read_u8()

                                    .context("read delta copy size byte fail")?;

                                copy_size |= (b as usize) << 16;

                            }

                            if copy_size == 0 {

                                copy_size = 0x1000;

                            }

  

                            let copy_end = copy_offset + copy_size;

                            if copy_end > src_size {

                                anyhow::bail!("copy end out of src size");

                            }

  

                            new_tgt.extend_from_slice(

                                &base_object.reader.get_ref()[copy_offset..copy_end],

                            ); // no-op, just for clarity

                        } else {

                            let size = (opcode & 0b0111_1111) as usize;

                            new_tgt.extend_from_slice(

                                &delta_data.get_ref()[delta_data.position() as usize

                                    ..delta_data.position() as usize + size],

                            );

                            delta_data.set_position(delta_data.position() as u64 + size as u64);

                        }

                    }

  

                    let mut object = Object {

                        kind: base_object.kind.clone(),

                        expected_size: tgt_size as u64,

                        reader: Cursor::new(new_tgt),

                    };

                    let hash = object.compute_hash(std::io::sink(), false).await?;

                    object.reader.set_position(0);

                    eprintln!(

                        "RefDelta Object {} is of type {:?} with hash {:?}",

                        object_index,

                        object.kind,

                        hex::encode(hash)

                    );

                    hashmap.insert(hash, object);

                }

```
- 读取hash，找到基本对象，以此为基础构造RefDelta
	- **复制：** 根据指定的偏移量和大小，从引用的对象复制数据。
		- 每个 `ADD` 命令的长度限制为 **127 字节，** 因为标签字节只有 7 位来编码长度。
	    - 对于较大的新增数据，将数据分成若干块，每块数据前面加上一个 `ADD` 标签。
	- **添加：** 将字面数据插入对象。
- 将构造后的对象放入map，作为后面对象构造的基础对象
### 3. 核心实现逻辑
- 解析引用列表中的head指向的commit的hash和分支，随后构造want have请求packfile
- 主要参考：
	- https://i27ae15.github.io/git-protocol-doc/docs/git-protocol/ref-delta
	- [Git-gitprotocol-pack - 3-如何通过网络传输软件包-packfile文件格式](../notes/Git-智能协议/Git-gitprotocol-pack%20-%203-如何通过网络传输软件包-packfile文件格式.md)
### 4. 测试
- [智能协议测试](../notes/Git-智能协议/智能协议测试.md)
### 5.总结
