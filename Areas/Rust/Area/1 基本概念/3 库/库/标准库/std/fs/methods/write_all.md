```
let mut file = File::create(&obj_path)?;
file.write_all(&zlib_content)?;
```
等价于
```
fs::write(path, &data)?;
```
小文件一次性写入