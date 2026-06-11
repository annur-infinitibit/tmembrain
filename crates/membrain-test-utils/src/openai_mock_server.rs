//! Scripted HTTP server for LLM integration tests.
//!
//! Not a mock framework. A real `tokio::net::TcpListener` serving pre-queued
//! HTTP responses (status + body) in FIFO order. Lives on `127.0.0.1:0` so
//! tests grab an ephemeral port.
//!
//! Use via:
//! ```no_run
//! # use membrain_test_utils::openai_mock_server::{ScriptedOpenAiServer, ScriptedResponse};
//! # async fn example() {
//! let server = ScriptedOpenAiServer::start().await;
//! server.queue_response(ScriptedResponse::ok_json("{\"decision\":\"add\"}"));
//! let base_url = server.chat_completions_base_url();
//! // call resolver configured with `base_url` ...
//! server.shutdown().await;
//! # }
//! ```

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// A scripted HTTP response.
#[derive(Debug, Clone)]
pub struct ScriptedResponse {
    /// HTTP status code (e.g. 200, 429, 500).
    pub status: u16,
    /// Optional delay applied before sending the response.
    pub delay: Option<Duration>,
    /// Response body bytes.
    pub body: Vec<u8>,
    /// Content-Type header value.
    pub content_type: String,
}

impl ScriptedResponse {
    /// 200 OK with JSON body.
    pub fn ok_json(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            delay: None,
            body: body.into().into_bytes(),
            content_type: "application/json".to_string(),
        }
    }

    /// Arbitrary status + body.
    pub fn status(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            delay: None,
            body: body.into().into_bytes(),
            content_type: "application/json".to_string(),
        }
    }

    /// Add an artificial delay before responding.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

/// Captured request observed by the server.
#[derive(Debug, Clone)]
pub struct CapturedRequest {
    /// Request method (GET/POST).
    pub method: String,
    /// Request target (path + query).
    pub target: String,
    /// All headers as `(lowercased_name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Decoded request body.
    pub body: Vec<u8>,
}

impl CapturedRequest {
    /// Find the first header by lowercase name.
    pub fn header(&self, name: &str) -> Option<&str> {
        let needle = name.to_lowercase();
        self.headers
            .iter()
            .find(|(header_name, _)| header_name == &needle)
            .map(|(_, value)| value.as_str())
    }

    /// Decode the body as UTF-8, lossy.
    pub fn body_string(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

#[derive(Default)]
struct SharedState {
    queue: VecDeque<ScriptedResponse>,
    captured: Vec<CapturedRequest>,
}

/// Scripted HTTP server instance.
pub struct ScriptedOpenAiServer {
    address: SocketAddr,
    state: Arc<Mutex<SharedState>>,
    shutdown: Arc<Notify>,
    handle: JoinHandle<()>,
}

impl ScriptedOpenAiServer {
    /// Bind to an ephemeral port and start accepting connections.
    pub async fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let state = Arc::new(Mutex::new(SharedState::default()));
        let shutdown = Arc::new(Notify::new());

        let accept_state = Arc::clone(&state);
        let accept_shutdown = Arc::clone(&shutdown);
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = accept_shutdown.notified() => break,
                    accepted = listener.accept() => {
                        match accepted {
                            Ok((stream, _)) => {
                                let conn_state = Arc::clone(&accept_state);
                                tokio::spawn(async move {
                                    let _ = handle_connection(stream, conn_state).await;
                                });
                            }
                            Err(_) => break,
                        }
                    }
                }
            }
        });

        Ok(Self {
            address,
            state,
            shutdown,
            handle,
        })
    }

    /// Listening socket address (e.g. `127.0.0.1:54321`).
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Base URL suitable for passing as `ConflictResolutionConfig::base_url`
    /// (trailing `/v1` added to match OpenAI's convention; the server ignores
    /// the path and only uses the queued responses).
    pub fn chat_completions_base_url(&self) -> String {
        format!("http://{}/v1", self.address)
    }

    /// Append a scripted response to the FIFO queue.
    pub fn queue_response(&self, response: ScriptedResponse) {
        self.state.lock().queue.push_back(response);
    }

    /// Snapshot of requests observed so far.
    pub fn captured_requests(&self) -> Vec<CapturedRequest> {
        self.state.lock().captured.clone()
    }

    /// Most recent captured request, or `None` if none have been observed.
    pub fn last_request(&self) -> Option<CapturedRequest> {
        self.state.lock().captured.last().cloned()
    }

    /// Stop accepting connections and abort the accept loop.
    pub async fn shutdown(self) {
        self.shutdown.notify_waiters();
        self.handle.abort();
        let _ = self.handle.await;
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<Mutex<SharedState>>,
) -> std::io::Result<()> {
    let mut buffer = vec![0_u8; 16_384];
    let mut accumulated = Vec::new();
    let mut header_end = None;
    while header_end.is_none() {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        accumulated.extend_from_slice(&buffer[..read]);
        header_end = find_header_end(&accumulated);
    }
    let split_at = header_end.unwrap_or(accumulated.len());
    let header_bytes = &accumulated[..split_at];
    let headers_text = String::from_utf8_lossy(header_bytes).into_owned();
    let mut lines = headers_text.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();

    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name_lower = name.trim().to_lowercase();
            let value = value.trim().to_string();
            if name_lower == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((name_lower, value));
        }
    }

    let mut body = accumulated.split_off(split_at);
    while body.len() < content_length {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&buffer[..read]);
    }
    body.truncate(content_length);

    let captured = CapturedRequest {
        method,
        target,
        headers,
        body,
    };

    let response = {
        let mut guard = state.lock();
        guard.captured.push(captured);
        guard.queue.pop_front().unwrap_or_else(|| ScriptedResponse {
            status: 500,
            delay: None,
            body: b"no scripted response queued".to_vec(),
            content_type: "text/plain".to_string(),
        })
    };

    if let Some(delay) = response.delay {
        tokio::time::sleep(delay).await;
    }

    let status_text = status_message(response.status);
    let reply = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {ct}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n",
        status = response.status,
        ct = response.content_type,
        length = response.body.len(),
    );
    stream.write_all(reply.as_bytes()).await?;
    stream.write_all(&response.body).await?;
    stream.flush().await?;
    Ok(())
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

fn status_message(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}
