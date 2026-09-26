//! Dependency-free Chrome DevTools Protocol client: a minimal RFC 6455 WebSocket over
//! `std::net::TcpStream` and JSON-RPC on top (`serde_json`).
//!
//! Only what a local CDP endpoint needs is implemented: `ws://` (no TLS), client→server
//! masking, fragmented messages, ping/pong and close.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use candle_core::{bail, Error, Result};
use serde_json::{json, Value};

const OP_CONTINUATION: u8 = 0x0;
const OP_TEXT: u8 = 0x1;
const OP_BINARY: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;
const OP_PING: u8 = 0x9;
const OP_PONG: u8 = 0xA;

/// Encodes one final frame. Client frames must be masked (`mask = Some(key)`).
pub fn encode_frame(opcode: u8, payload: &[u8], mask: Option<[u8; 4]>) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 14);
    out.push(0x80 | opcode);
    let mask_bit = if mask.is_some() { 0x80 } else { 0 };
    match payload.len() {
        n if n < 126 => out.push(mask_bit | n as u8),
        n if n <= u16::MAX as usize => {
            out.push(mask_bit | 126);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            out.push(mask_bit | 127);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    match mask {
        Some(key) => {
            out.extend_from_slice(&key);
            out.extend(payload.iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
        }
        None => out.extend_from_slice(payload),
    }
    out
}

/// One decoded frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub fin: bool,
    pub opcode: u8,
    pub payload: Vec<u8>,
}

/// Reads the rest of a frame whose first byte is `first`.
pub fn decode_frame_after(first: u8, r: &mut impl Read) -> std::io::Result<Frame> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    let masked = b[0] & 0x80 != 0;
    let len = match b[0] & 0x7F {
        126 => {
            let mut l = [0u8; 2];
            r.read_exact(&mut l)?;
            u16::from_be_bytes(l) as usize
        }
        127 => {
            let mut l = [0u8; 8];
            r.read_exact(&mut l)?;
            u64::from_be_bytes(l) as usize
        }
        n => n as usize,
    };
    let mut key = [0u8; 4];
    if masked {
        r.read_exact(&mut key)?;
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    if masked {
        payload.iter_mut().enumerate().for_each(|(i, b)| *b ^= key[i % 4]);
    }
    Ok(Frame { fin: first & 0x80 != 0, opcode: first & 0x0F, payload })
}

/// A client WebSocket connection.
pub struct WebSocket {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    mask_state: u32,
}

impl WebSocket {
    /// Connects to `ws://host:port/path` and performs the opening handshake.
    pub fn connect(url: &str) -> Result<Self> {
        let Some(rest) = url.strip_prefix("ws://") else { bail!("only ws:// URLs are supported, got {url}") };
        let (host, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
        let path = if path.is_empty() { "/" } else { path };
        let mut writer = TcpStream::connect(host).map_err(Error::wrap)?;
        writer.set_nodelay(true).map_err(Error::wrap)?;
        writer.set_read_timeout(Some(Duration::from_secs(30))).map_err(Error::wrap)?;
        let mut reader = BufReader::new(writer.try_clone().map_err(Error::wrap)?);
        write!(
            writer,
            "GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        .map_err(Error::wrap)?;
        let mut status = String::new();
        reader.read_line(&mut status).map_err(Error::wrap)?;
        if !status.contains(" 101 ") {
            bail!("websocket handshake with {url} failed: {}", status.trim())
        }
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(Error::wrap)? > 2 {
            line.clear();
        }
        // Frames that arrived with the handshake stay in `reader`'s buffer.
        Ok(Self { reader, writer, mask_state: 0x2545_F491 })
    }

    fn next_mask(&mut self) -> [u8; 4] {
        // xorshift32: masking only has to be unpredictable to intermediaries, not secret.
        let mut x = self.mask_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.mask_state = x;
        x.to_le_bytes()
    }

    fn send_frame(&mut self, opcode: u8, payload: &[u8]) -> Result<()> {
        let mask = self.next_mask();
        self.writer.write_all(&encode_frame(opcode, payload, Some(mask))).map_err(Error::wrap)
    }

    pub fn send_text(&mut self, text: &str) -> Result<()> {
        self.send_frame(OP_TEXT, text.as_bytes())
    }

    /// Next text message, or `None` if nothing started arriving before `deadline`.
    pub fn recv_text(&mut self, deadline: Instant) -> Result<Option<String>> {
        let mut message: Vec<u8> = Vec::new();
        loop {
            let first = if message.is_empty() {
                match self.first_byte(deadline)? {
                    Some(b) => b,
                    None => return Ok(None),
                }
            } else {
                // Inside a fragmented message: the rest is already on its way.
                self.first_byte(Instant::now() + Duration::from_secs(30))?
                    .ok_or_else(|| Error::Msg("websocket: truncated fragmented message".into()))?
            };
            self.writer.set_read_timeout(Some(Duration::from_secs(30))).map_err(Error::wrap)?;
            let frame = decode_frame_after(first, &mut self.reader).map_err(Error::wrap)?;
            match frame.opcode {
                OP_PING => self.send_frame(OP_PONG, &frame.payload)?,
                OP_PONG => {}
                OP_CLOSE => bail!("websocket closed by the browser"),
                OP_TEXT | OP_BINARY | OP_CONTINUATION => {
                    message.extend_from_slice(&frame.payload);
                    if frame.fin {
                        return String::from_utf8(message).map(Some).map_err(Error::wrap);
                    }
                }
                op => bail!("websocket: unexpected opcode {op:#x}"),
            }
        }
    }

    fn first_byte(&mut self, deadline: Instant) -> Result<Option<u8>> {
        let mut b = [0u8; 1];
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() && self.reader.buffer().is_empty() {
                return Ok(None);
            }
            self.writer.set_read_timeout(Some(left.max(Duration::from_millis(1)))).map_err(Error::wrap)?;
            match self.reader.read(&mut b) {
                Ok(1) => return Ok(Some(b[0])),
                Ok(_) => bail!("websocket: connection closed"),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    if Instant::now() >= deadline {
                        return Ok(None);
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(Error::wrap(e)),
            }
        }
    }
}

/// A CDP session over one WebSocket (a page target).
pub struct Cdp {
    ws: WebSocket,
    next_id: u64,
    events: VecDeque<Value>,
    /// Timeout of a single command.
    pub timeout: Duration,
}

impl Cdp {
    pub fn connect(ws_url: &str) -> Result<Self> {
        Ok(Self {
            ws: WebSocket::connect(ws_url)?,
            next_id: 1,
            events: VecDeque::new(),
            timeout: Duration::from_secs(30),
        })
    }

    /// Sends `method(params)` and waits for its result; events received meanwhile are queued.
    pub fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.ws.send_text(&json!({ "id": id, "method": method, "params": params }).to_string())?;
        let deadline = Instant::now() + self.timeout;
        loop {
            let Some(text) = self.ws.recv_text(deadline)? else {
                bail!("CDP {method}: no reply in {:?}", self.timeout)
            };
            let msg: Value = serde_json::from_str(&text).map_err(Error::wrap)?;
            if msg.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(err) = msg.get("error") {
                    bail!("CDP {method}: {err}")
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            if msg.get("method").is_some() {
                self.events.push_back(msg);
            }
        }
    }

    /// Forgets queued events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Waits for the first event whose method is in `methods` (queued events count) and
    /// returns its method name, or `None` after `timeout`. Earlier events are dropped.
    pub fn wait_event(&mut self, methods: &[&str], timeout: Duration) -> Result<Option<String>> {
        let is_wanted =
            |m: &Value| m.get("method").and_then(Value::as_str).filter(|s| methods.contains(s)).map(str::to_string);
        while let Some(m) = self.events.pop_front() {
            if let Some(name) = is_wanted(&m) {
                return Ok(Some(name));
            }
        }
        let deadline = Instant::now() + timeout;
        while let Some(text) = self.ws.recv_text(deadline)? {
            let msg: Value = serde_json::from_str(&text).map_err(Error::wrap)?;
            if let Some(name) = is_wanted(&msg) {
                return Ok(Some(name));
            }
        }
        Ok(None)
    }

    /// Evaluates a JavaScript expression in the page and returns its value (JSON).
    pub fn eval(&mut self, expression: &str) -> Result<Value> {
        let r = self.call("Runtime.evaluate", json!({ "expression": expression, "returnByValue": true }))?;
        if let Some(ex) = r.get("exceptionDetails") {
            bail!("javascript error: {}", ex.get("text").and_then(Value::as_str).unwrap_or("?"))
        }
        Ok(r.pointer("/result/value").cloned().unwrap_or(Value::Null))
    }
}

/// Plain `GET` of a local HTTP endpoint (the `/json/*` CDP discovery API), returns the body.
/// Chromium keeps the connection open, so the body is read by `Content-Length`.
pub fn http_get(host: &str, path: &str) -> Result<String> {
    let mut s = TcpStream::connect(host).map_err(Error::wrap)?;
    s.set_read_timeout(Some(Duration::from_secs(10))).map_err(Error::wrap)?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").map_err(Error::wrap)?;
    let mut r = BufReader::new(s);
    let mut status = String::new();
    r.read_line(&mut status).map_err(Error::wrap)?;
    let mut len = None;
    let mut line = String::new();
    while r.read_line(&mut line).map_err(Error::wrap)? > 2 {
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse::<usize>().ok();
            }
        }
        line.clear();
    }
    if !status.contains(" 200 ") {
        bail!("GET {host}{path} failed: {}", status.trim())
    }
    let mut body = Vec::new();
    match len {
        Some(n) => {
            body.resize(n, 0);
            r.read_exact(&mut body).map_err(Error::wrap)?;
        }
        None => {
            r.read_to_end(&mut body).map_err(Error::wrap)?;
        }
    }
    String::from_utf8(body).map_err(Error::wrap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip() {
        for len in [0usize, 5, 125, 126, 300, 65535, 65536, 70000] {
            let payload: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            for mask in [None, Some([1, 2, 3, 4])] {
                let bytes = encode_frame(OP_TEXT, &payload, mask);
                let mut r = &bytes[1..];
                let f = decode_frame_after(bytes[0], &mut r).unwrap();
                assert_eq!(f, Frame { fin: true, opcode: OP_TEXT, payload: payload.clone() });
                assert!(r.is_empty());
            }
        }
        // RFC 6455 §5.7: unmasked "Hello"
        assert_eq!(encode_frame(OP_TEXT, b"Hello", None), [0x81, 0x05, 0x48, 0x65, 0x6c, 0x6c, 0x6f]);
        // …and the masked one
        let masked = encode_frame(OP_TEXT, b"Hello", Some([0x37, 0xfa, 0x21, 0x3d]));
        assert_eq!(masked, [0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58]);
    }
}
