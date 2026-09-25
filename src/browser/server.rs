//! Minimal HTTP/1.1 server for the sandbox web (one thread per connection, `GET` only).

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use candle_core::{Error, Result};

use super::world::{Page, World};

/// Serves [`World`] pages until dropped.
pub struct SiteServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl SiteServer {
    /// Binds `addr` (use port 0 for a free port) and starts serving in the background.
    pub fn start(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(Error::wrap)?;
        let addr = listener.local_addr().map_err(Error::wrap)?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = std::thread::spawn(move || {
            for conn in listener.incoming() {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(stream) = conn {
                    std::thread::spawn(move || {
                        let _ = handle(stream);
                    });
                }
            }
        });
        Ok(Self { addr, stop, thread: Some(thread) })
    }

    /// `http://host:port`.
    pub fn origin(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Serves until the process is killed.
    pub fn wait(mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for SiteServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wake the accept loop so it observes the flag.
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn handle(stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    // Skip the headers.
    let mut line = String::new();
    while reader.read_line(&mut line)? > 2 {
        line.clear();
    }
    let mut parts = request_line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    let (status, body) = match (method, World::page_at(path)) {
        ("GET", Page::NotFound) => ("404 Not Found", Page::NotFound.html()),
        ("GET", page) => ("200 OK", page.html()),
        _ => ("405 Method Not Allowed", String::new()),
    };
    let mut out = stream;
    write!(
        out,
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn get(server: &SiteServer, path: &str) -> String {
        let mut s = TcpStream::connect(server.addr()).unwrap();
        write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn serves_pages() {
        let server = SiteServer::start("127.0.0.1:0").unwrap();
        let home = get(&server, "/w/3/");
        assert!(home.starts_with("HTTP/1.1 200 OK"));
        assert!(home.contains("<input name=\"q\""));
        let item = get(&server, "/w/3/item/lamp");
        assert!(item.contains("<h1>lamp</h1>") && item.contains("<th>price</th>"));
        assert!(get(&server, "/nope").starts_with("HTTP/1.1 404"));
    }
}
