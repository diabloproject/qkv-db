use std::env::args;
use std::io::{Read, stdin, Write};
use std::net::TcpStream;

fn main() {
    let args: Vec<String> = args().collect();
    if let Some(path) = args.get(1) {
        let mut stream = TcpStream::connect("127.0.0.1:7878").unwrap();
        let buf = std::fs::read(path).unwrap();
        stream.write(&(buf.len() as u32).to_le_bytes()).unwrap();
        stream.write_all(&buf).unwrap();
        stream.flush().unwrap();

        println!("Command sent.");
        let mut content_size = [0u8; 4];
        stream.read_exact(&mut content_size).unwrap();
        let content_size = u32::from_le_bytes(content_size);
        println!("{content_size}");
        let mut content = Vec::from_iter((0..content_size).map(|_| 0u8));
        stream.read_exact(&mut content).unwrap();
        let content = String::from_utf8(content).unwrap();
        println!("{content}")
    }
    loop {
        let mut stream = TcpStream::connect("127.0.0.1:7878").unwrap();
        let mut buf = String::new();
        print!("> ");
        std::io::stdout().flush().unwrap();
        stdin().read_line(&mut buf).unwrap();
        stream.write(&(buf.as_bytes().len() as u32).to_le_bytes()).unwrap();
        stream.write_all(buf.as_bytes()).unwrap();
        stream.flush().unwrap();

        println!("Command sent.");
        buf.clear();
        let mut content_size = [0u8; 4];
        stream.read_exact(&mut content_size).unwrap();
        let content_size = u32::from_le_bytes(content_size);
        let mut content = Vec::from_iter((0..content_size).map(|_| 0u8));
        stream.read_exact(&mut content).unwrap();
        let content = String::from_utf8(content).unwrap();
        println!("{content}")
    }
}
