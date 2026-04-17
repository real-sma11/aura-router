//! Provider-aware SSE proxy with billing capture.
//!
//! Anthropic streams are passed through as-is while we parse usage metadata.
//! OpenAI chat-completion streams are translated into Anthropic-style `/v1/messages`
//! events so Aura clients can keep speaking one hosted protocol.

use std::collections::{BTreeMap, VecDeque};

use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::providers::{self, Provider};

/// Token usage extracted from an SSE stream.
#[derive(Debug, Default)]
pub struct StreamUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub model: Option<String>,
}

/// Proxy an SSE stream from the upstream provider to the client,
/// capturing billing data along the way.
pub fn proxy_stream(
    provider: Provider,
    requested_model: &str,
    upstream: reqwest::Response,
) -> (
    impl futures_util::Stream<Item = Result<Bytes, reqwest::Error>>,
    oneshot::Receiver<StreamUsage>,
) {
    let (usage_tx, usage_rx) = oneshot::channel();
    let byte_stream = upstream.bytes_stream();

    let tee_stream = TeeStream {
        inner: Box::pin(byte_stream),
        adapter: StreamAdapter::new(provider, requested_model),
        usage_tx: Some(usage_tx),
        finished: false,
        pending_output: VecDeque::new(),
    };

    (tee_stream, usage_rx)
}

/// A stream that either forwards raw Anthropic SSE bytes or emits translated
/// Anthropic-compatible SSE for OpenAI-backed models.
struct TeeStream<S> {
    inner: std::pin::Pin<Box<S>>,
    adapter: StreamAdapter,
    usage_tx: Option<oneshot::Sender<StreamUsage>>,
    finished: bool,
    pending_output: VecDeque<Bytes>,
}

impl<S, E> futures_util::Stream for TeeStream<S>
where
    S: futures_util::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<Bytes, E>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if let Some(bytes) = self.pending_output.pop_front() {
            return std::task::Poll::Ready(Some(Ok(bytes)));
        }

        if self.finished {
            return std::task::Poll::Ready(None);
        }

        loop {
            match self.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    let mut emitted = VecDeque::new();
                    self.adapter.feed(bytes, &mut emitted);
                    self.pending_output.extend(emitted);
                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return std::task::Poll::Ready(Some(Err(e)));
                }
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    let mut emitted = VecDeque::new();
                    let usage = self.adapter.finish(&mut emitted);
                    self.pending_output.extend(emitted);
                    let model = usage.model.as_deref().unwrap_or("unknown");
                    let max_tokens = providers::max_context_tokens(model);
                    let context_usage = if max_tokens > 0 {
                        usage.input_tokens as f64 / max_tokens as f64
                    } else {
                        0.0
                    };

                    self.pending_output.push_back(Bytes::from(format!(
                        "event: x_context_usage\ndata: {{\"contextUsage\":{context_usage:.4},\"inputTokens\":{},\"outputTokens\":{},\"maxTokens\":{max_tokens}}}\n\n",
                        usage.input_tokens, usage.output_tokens
                    )));

                    if let Some(tx) = self.usage_tx.take() {
                        let _ = tx.send(usage);
                    }

                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }

                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    if let Some(next) = self.pending_output.pop_front() {
                        return std::task::Poll::Ready(Some(Ok(next)));
                    }
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

enum StreamAdapter {
    Anthropic(AnthropicPassthrough),
    OpenAi(OpenAiCompatStream),
}

impl StreamAdapter {
    fn new(provider: Provider, requested_model: &str) -> Self {
        match provider {
            Provider::Anthropic => Self::Anthropic(AnthropicPassthrough {
                parser: SseParser::new(),
            }),
            Provider::OpenAi | Provider::Fireworks => {
                Self::OpenAi(OpenAiCompatStream::new(requested_model))
            }
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        match self {
            Self::Anthropic(adapter) => adapter.feed(bytes, output),
            Self::OpenAi(adapter) => adapter.feed(bytes, output),
        }
    }

    fn finish(&mut self, output: &mut VecDeque<Bytes>) -> StreamUsage {
        match self {
            Self::Anthropic(adapter) => adapter.finish(output),
            Self::OpenAi(adapter) => adapter.finish(output),
        }
    }
}

struct AnthropicPassthrough {
    parser: SseParser,
}

impl AnthropicPassthrough {
    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        self.parser.feed(&bytes);
        output.push_back(bytes);
    }

    fn finish(&mut self, _output: &mut VecDeque<Bytes>) -> StreamUsage {
        self.parser.finalize()
    }
}

struct OpenAiCompatStream {
    requested_model: String,
    buffer: String,
    usage: StreamUsage,
    message_id: Option<String>,
    message_started: bool,
    text_block_index: Option<usize>,
    text_block_open: bool,
    tool_blocks: BTreeMap<usize, ToolBlockState>,
    pending_stop_reason: Option<String>,
    finalized: bool,
}

#[derive(Default)]
struct ToolBlockState {
    content_index: usize,
    id: String,
    name: String,
    started: bool,
    pending_json: String,
    stopped: bool,
}

impl OpenAiCompatStream {
    fn new(requested_model: &str) -> Self {
        Self {
            requested_model: requested_model.to_string(),
            buffer: String::new(),
            usage: StreamUsage {
                model: Some(requested_model.to_string()),
                ..StreamUsage::default()
            },
            message_id: None,
            message_started: false,
            text_block_index: None,
            text_block_open: false,
            tool_blocks: BTreeMap::new(),
            pending_stop_reason: None,
            finalized: false,
        }
    }

    fn feed(&mut self, bytes: Bytes, output: &mut VecDeque<Bytes>) {
        let text = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&text);

        while let Some((event, delimiter_len)) = next_sse_event(&self.buffer) {
            let event_str = self.buffer[..event].to_string();
            self.buffer = self.buffer[event + delimiter_len..].to_string();
            self.process_event(&event_str, output);
        }
    }

    fn finish(&mut self, output: &mut VecDeque<Bytes>) -> StreamUsage {
        self.finish_message(output);
        StreamUsage {
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            cache_creation_input_tokens: self.usage.cache_creation_input_tokens,
            cache_read_input_tokens: self.usage.cache_read_input_tokens,
            model: self.usage.model.clone(),
        }
    }

    fn process_event(&mut self, event_str: &str, output: &mut VecDeque<Bytes>) {
        let mut data_lines = Vec::new();

        for line in event_str.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data.to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }

        if data_lines.is_empty() {
            return;
        }

        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.finish_message(output);
            return;
        }

        let Ok(chunk) = serde_json::from_str::<Value>(&data) else {
            output.push_back(anthropic_sse(
                "error",
                json!({
                    "type": "error",
                    "error": {
                        "message": "malformed OpenAI SSE JSON"
                    }
                }),
            ));
            self.finalized = true;
            return;
        };

        if let Some(id) = chunk.get("id").and_then(Value::as_str) {
            self.message_id.get_or_insert_with(|| id.to_string());
        }

        if let Some(model) = chunk.get("model").and_then(Value::as_str) {
            self.usage.model = Some(model.to_string());
        }

        if let Some(usage) = chunk.get("usage").and_then(Value::as_object) {
            self.usage.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.input_tokens);
            self.usage.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.output_tokens);
            if self.pending_stop_reason.is_some() {
                self.finish_message(output);
            }
        }

        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            return;
        };

        if !choices.is_empty() {
            self.ensure_message_started(output);
        }

        for choice in choices {
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        self.ensure_text_block_started(output);
                        if let Some(index) = self.text_block_index {
                            output.push_back(anthropic_sse(
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content,
                                    }
                                }),
                            ));
                        }
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        self.process_tool_call_delta(tool_call, output);
                    }
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.pending_stop_reason = Some(finish_reason.to_string());
                if self.usage.output_tokens > 0 || finish_reason == "tool_calls" {
                    self.finish_message(output);
                }
            }
        }
    }

    fn process_tool_call_delta(&mut self, tool_call: &Value, output: &mut VecDeque<Bytes>) {
        let tool_index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let content_index = self.tool_content_index(tool_index);
        let state = self
            .tool_blocks
            .entry(tool_index)
            .or_insert_with(|| ToolBlockState {
                content_index,
                ..ToolBlockState::default()
            });

        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            state.id = id.to_string();
        }

        if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str) {
            if state.name.is_empty() {
                state.name = name.to_string();
            } else if !name.is_empty() && !state.name.ends_with(name) {
                state.name.push_str(name);
            }
        }

        if !state.started && !state.id.is_empty() && !state.name.is_empty() {
            output.push_back(anthropic_sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": state.content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.id,
                        "name": state.name,
                        "input": {},
                    }
                }),
            ));
            state.started = true;

            if !state.pending_json.is_empty() {
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": state.content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": state.pending_json,
                        }
                    }),
                ));
                state.pending_json.clear();
            }
        }

        if let Some(arguments) = tool_call.pointer("/function/arguments").and_then(Value::as_str) {
            if state.started {
                output.push_back(anthropic_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": state.content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        }
                    }),
                ));
            } else {
                state.pending_json.push_str(arguments);
            }
        }
    }

    fn ensure_message_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.message_started {
            return;
        }
        self.message_started = true;

        let message_id = self
            .message_id
            .clone()
            .unwrap_or_else(|| format!("msg_{}", Uuid::new_v4().simple()));

        output.push_back(anthropic_sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.requested_model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": self.usage.input_tokens,
                    }
                }
            }),
        ));
    }

    fn ensure_text_block_started(&mut self, output: &mut VecDeque<Bytes>) {
        if self.text_block_open {
            return;
        }

        let index = if let Some(index) = self.text_block_index {
            index
        } else if self.tool_blocks.is_empty() {
            0
        } else {
            self.tool_blocks
                .values()
                .map(|state| state.content_index)
                .max()
                .unwrap_or(0)
                + 1
        };

        self.text_block_index = Some(index);
        self.text_block_open = true;
        output.push_back(anthropic_sse(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": "",
                }
            }),
        ));
    }

    fn tool_content_index(&self, tool_index: usize) -> usize {
        if self.text_block_index == Some(0) {
            tool_index + 1
        } else {
            tool_index
        }
    }

    fn finish_message(&mut self, output: &mut VecDeque<Bytes>) {
        if self.finalized || !self.message_started {
            return;
        }

        if self.text_block_open {
            if let Some(index) = self.text_block_index {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                ));
            }
            self.text_block_open = false;
        }

        for state in self.tool_blocks.values_mut() {
            if state.started && !state.stopped {
                output.push_back(anthropic_sse(
                    "content_block_stop",
                    json!({
                        "type": "content_block_stop",
                        "index": state.content_index,
                    }),
                ));
                state.stopped = true;
            }
        }

        output.push_back(anthropic_sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_openai_finish_reason(self.pending_stop_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "output_tokens": self.usage.output_tokens,
                }
            }),
        ));
        output.push_back(anthropic_sse(
            "message_stop",
            json!({
                "type": "message_stop",
            }),
        ));
        self.finalized = true;
    }
}

/// Parses Anthropic SSE lines to extract billing-relevant data.
struct SseParser {
    buffer: String,
    current_event: Option<String>,
    usage: StreamUsage,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event: None,
            usage: StreamUsage::default(),
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();
            self.process_line(&line);
        }
    }

    fn process_line(&mut self, line: &str) {
        if let Some(event_type) = line.strip_prefix("event: ") {
            self.current_event = Some(event_type.to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            if let Some(ref event_type) = self.current_event.clone() {
                self.process_event(event_type, data);
            }
        } else if line.is_empty() {
            self.current_event = None;
        }
    }

    fn process_event(&mut self, event_type: &str, data: &str) {
        match event_type {
            "message_start" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(usage) = value.pointer("/message/usage") {
                        if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                            self.usage.input_tokens = n;
                        }
                        if let Some(n) = usage
                            .get("cache_creation_input_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            self.usage.cache_creation_input_tokens = n;
                        }
                        if let Some(n) = usage
                            .get("cache_read_input_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            self.usage.cache_read_input_tokens = n;
                        }
                    }
                    if let Some(model) = value.pointer("/message/model").and_then(|v| v.as_str()) {
                        self.usage.model = Some(model.to_string());
                    }
                }
            }
            "message_delta" => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(n) = value
                        .pointer("/usage/output_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        self.usage.output_tokens = n;
                    }
                }
            }
            _ => {}
        }
    }

    fn finalize(&self) -> StreamUsage {
        StreamUsage {
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            cache_creation_input_tokens: self.usage.cache_creation_input_tokens,
            cache_read_input_tokens: self.usage.cache_read_input_tokens,
            model: self.usage.model.clone(),
        }
    }
}

fn next_sse_event(buffer: &str) -> Option<(usize, usize)> {
    let event_end = buffer.find("\n\n").or_else(|| buffer.find("\r\n\r\n"))?;
    let delimiter_len = if buffer[event_end..].starts_with("\r\n\r\n") {
        4
    } else {
        2
    };
    Some((event_end, delimiter_len))
}

fn anthropic_sse(event_type: &str, payload: Value) -> Bytes {
    Bytes::from(format!("event: {event_type}\ndata: {payload}\n\n"))
}

fn map_openai_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        Some("stop") => "end_turn",
        Some("content_filter") => "end_turn",
        _ => "end_turn",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn bytes_stream(
        chunks: Vec<&'static str>,
    ) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Unpin {
        futures_util::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok(bytes::Bytes::from(c.to_string()))),
        )
    }

    #[tokio::test]
    async fn anthropic_passthrough_preserves_stream_and_usage() {
        let stream = bytes_stream(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12},\"model\":\"aura-claude-sonnet-4-6\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(Provider::Anthropic, "aura-claude-sonnet-4-6"),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
        };

        let mut seen = Vec::new();
        while let Some(chunk) = tee.next().await {
            seen.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert!(seen.iter().any(|chunk| chunk.contains("message_start")));
        assert!(seen.iter().any(|chunk| chunk.contains("x_context_usage")));
    }

    #[tokio::test]
    async fn openai_text_stream_translates_to_anthropic_events() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-123\",\"model\":\"gpt-4.1\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(Provider::OpenAi, "aura-gpt-4.1"),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 5);
        assert!(emitted.iter().any(|chunk| chunk.contains("event: message_start")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"text\":\"Hello\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"text\":\" world\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"stop_reason\":\"end_turn\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("event: message_stop")));
    }

    #[tokio::test]
    async fn openai_tool_call_stream_translates_to_tool_use_blocks() {
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"aura\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"chatcmpl-456\",\"model\":\"gpt-4.1\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}\n\n",
            "data: [DONE]\n\n",
        ]);
        let (tx, rx) = oneshot::channel();
        let mut tee = TeeStream {
            inner: Box::pin(stream),
            adapter: StreamAdapter::new(Provider::OpenAi, "aura-gpt-4.1"),
            usage_tx: Some(tx),
            finished: false,
            pending_output: VecDeque::new(),
        };

        let mut emitted = Vec::new();
        while let Some(chunk) = tee.next().await {
            emitted.push(String::from_utf8_lossy(&chunk.unwrap()).to_string());
        }

        let usage = rx.await.unwrap();
        assert_eq!(usage.output_tokens, 4);
        assert!(emitted.iter().any(|chunk| chunk.contains("\"type\":\"tool_use\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"partial_json\":\"{\\\"q\\\":\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"partial_json\":\"\\\"aura\\\"}\"")));
        assert!(emitted.iter().any(|chunk| chunk.contains("\"stop_reason\":\"tool_use\"")));
    }
}
