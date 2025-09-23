```
commit {size}\0{content}
```

内容
```
tree {tree_sha}
{parents}
author {author_name} <{author_email}> {author_date_seconds} {author_date_timezone}
committer {committer_name} <{committer_email}> {committer_date_seconds} {committer_date_timezone}

{commit message}
```
{author_name}
`{author_email}`：例如：`cirosantilli@mail.com`。
`{author_date_seconds}`：自 1970 年以来的秒数，例如 `946684800` 是 2000 年的第一秒
`{author_date_timezone}`：例如：+`0000` 是 UTC