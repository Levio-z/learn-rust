byteorder
```rust
use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Result};

#[derive(Debug)]
pub struct Request {
    pub request_api_key: i16,
    pub request_api_version: i16,
    pub correlation_id: i32,
    // pub client_id: Option<String>,
    // tag_buffer: Vec<TaggedField>,
}

impl Request {
    pub fn parse_and_new(buffer: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(buffer);

        let message_size = cursor.read_i32::<BigEndian>()?;
        let request_api_key = cursor.read_i16::<BigEndian>()?;
        let request_api_version = cursor.read_i16::<BigEndian>()?;
        let correlation_id = cursor.read_i32::<BigEndian>()?;

        Ok(Request {
            request_api_key,
            request_api_version,
            correlation_id,
            // client_id,
            // tag_buffer,
        })
    }
}
```