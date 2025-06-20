```rust
     let size = std::cmp::min(

            buf.len(),

            self.tcp.header_len() as usize + self.ip.header_len() as usize + payload.len(),

        );

        self.ip

            .set_payload_len(size - self.ip.header_len() as usize);

```