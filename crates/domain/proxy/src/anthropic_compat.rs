use serde_json::{json, Value};

use crate::providers::Provider;

pub fn validate_request(provider: Provider, request: &Value) -> Result<(), String> {
    match provider {
        Provider::Anthropic => Ok(()),
        Provider::OpenAi => validate_openai_request(request),
        Provider::Fireworks => {
            validate_openai_request(request)?;
            validate_fireworks_privacy_policy(request)
        }
    }
}

pub fn request_to_upstream(
    provider: Provider,
    upstream_model: &str,
    request: &Value,
) -> Result<Value, String> {
    match provider {
        Provider::Anthropic => {
            let mut next = request.clone();
            next["model"] = Value::String(upstream_model.to_string());
            Ok(next)
        }
        Provider::OpenAi | Provider::Fireworks => {
            validate_openai_request(request)?;
            anthropic_request_to_openai(request, upstream_model)
        }
    }
}

pub fn response_from_upstream(
    provider: Provider,
    requested_model: &str,
    response: &Value,
) -> Result<Value, String> {
    match provider {
        Provider::Anthropic => {
            let mut next = response.clone();
            next["model"] = Value::String(requested_model.to_string());
            Ok(next)
        }
        Provider::OpenAi | Provider::Fireworks => {
            openai_response_to_anthropic(response, requested_model)
        }
    }
}

pub fn extract_last_user_text(request: &Value) -> String {
    request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .rfind(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .map(|message| flatten_anthropic_content_text(message.get("content")))
        .unwrap_or_default()
}

pub fn extract_response_text(response: &Value) -> String {
    flatten_anthropic_content_text(response.get("content"))
}

fn anthropic_request_to_openai(request: &Value, upstream_model: &str) -> Result<Value, String> {
    let mut messages = Vec::new();

    if let Some(system) = request.get("system") {
        let system_text = flatten_text_blocks(system);
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text,
            }));
        }
    }

    let request_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    for message in request_messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "Anthropic message is missing role".to_string())?;
        let blocks = message_content_blocks(message.get("content"))?;

        match role {
            "user" => append_user_messages(&blocks, &mut messages),
            "assistant" => messages.push(build_assistant_message(&blocks)),
            other => return Err(format!("Unsupported Anthropic role `{other}`")),
        }
    }

    let mut upstream = json!({
        "model": upstream_model,
        "messages": messages,
    });

    if let Some(max_tokens) = request.get("max_tokens").and_then(Value::as_u64) {
        upstream["max_completion_tokens"] = Value::from(max_tokens);
    }

    if let Some(temperature) = request.get("temperature").and_then(Value::as_f64) {
        upstream["temperature"] = Value::from(temperature);
    }

    if let Some(top_p) = request.get("top_p").and_then(Value::as_f64) {
        upstream["top_p"] = Value::from(top_p);
    }

    if let Some(stop_sequences) = request.get("stop_sequences").and_then(Value::as_array) {
        let stops = stop_sequences
            .iter()
            .filter_map(Value::as_str)
            .map(|value| Value::String(value.to_string()))
            .collect::<Vec<_>>();
        if !stops.is_empty() {
            upstream["stop"] = Value::Array(stops);
        }
    }

    let is_streaming = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if is_streaming {
        upstream["stream"] = Value::Bool(true);
        let mut stream_options = request
            .get("stream_options")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        stream_options.insert("include_usage".to_string(), Value::Bool(true));
        upstream["stream_options"] = Value::Object(stream_options);
    }

    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            upstream["tools"] = Value::Array(
                tools.iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.get("name").and_then(Value::as_str).unwrap_or_default(),
                                "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
                                "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                            }
                        })
                    })
                    .collect(),
            );
        }
    }

    if let Some(tool_choice) = request.get("tool_choice").and_then(Value::as_object) {
        let mapped = match tool_choice.get("type").and_then(Value::as_str) {
            Some("auto") => Some(Value::String("auto".to_string())),
            Some("any") => Some(Value::String("required".to_string())),
            Some("tool") => tool_choice.get("name").and_then(Value::as_str).map(|name| {
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            }),
            Some("none") => Some(Value::String("none".to_string())),
            _ => None,
        };
        if let Some(value) = mapped {
            upstream["tool_choice"] = value;
        }
    }

    Ok(upstream)
}

fn validate_openai_request(request: &Value) -> Result<(), String> {
    if request.get("thinking").is_some() {
        return Err(
            "OpenAI-backed Aura models do not support Anthropic thinking on /v1/messages yet"
                .to_string(),
        );
    }

    validate_system_blocks(request.get("system"))?;
    validate_message_blocks(request.get("messages"))?;

    Ok(())
}

fn validate_fireworks_privacy_policy(request: &Value) -> Result<(), String> {
    if request.get("store").is_some() {
        return Err(
            "Fireworks-backed Aura models do not allow request-level persistence controls on /v1/messages; Aura Router enforces the non-stored inference path centrally"
                .to_string(),
        );
    }

    if request.get("response_id").is_some() {
        return Err(
            "Fireworks-backed Aura models do not support response-state replay on /v1/messages; Aura Router uses stateless inference for privacy"
                .to_string(),
        );
    }

    Ok(())
}

fn validate_system_blocks(system: Option<&Value>) -> Result<(), String> {
    let Some(system) = system else {
        return Ok(());
    };

    match system {
        Value::String(_) => Ok(()),
        Value::Array(blocks) => {
            for block in blocks {
                let block_type = block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if block_type != "text" {
                    return Err(format!(
                        "OpenAI-backed Aura models only support text system blocks on /v1/messages (got `{block_type}`)"
                    ));
                }
            }
            Ok(())
        }
        _ => Err("System prompt must be a string or content block array".to_string()),
    }
}

fn validate_message_blocks(messages: Option<&Value>) -> Result<(), String> {
    let messages = messages
        .and_then(Value::as_array)
        .ok_or_else(|| "Anthropic request is missing messages array".to_string())?;

    for (message_index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Anthropic message {message_index} is missing role"))?;
        if !matches!(role, "user" | "assistant") {
            return Err(format!(
                "OpenAI-backed Aura models do not support Anthropic role `{role}`"
            ));
        }
        let blocks = match message.get("content") {
            Some(Value::String(_)) => continue,
            Some(Value::Array(blocks)) => blocks,
            Some(_) => {
                return Err(format!(
                    "Anthropic message {message_index} content must be a string or content array"
                ));
            }
            None => {
                return Err(format!(
                    "Anthropic message {message_index} is missing content"
                ));
            }
        };

        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let supported = match role {
                "user" => matches!(block_type, "text" | "image" | "tool_result"),
                "assistant" => matches!(block_type, "text" | "tool_use"),
                _ => false,
            };

            if !supported {
                return Err(format!(
                    "OpenAI-backed Aura models do not support `{block_type}` blocks for `{role}` messages on /v1/messages"
                ));
            }
        }
    }

    Ok(())
}

fn append_user_messages(blocks: &[Value], messages: &mut Vec<Value>) {
    if blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
    {
        append_mixed_user_and_tool_result_messages(blocks, messages);
        return;
    }

    append_non_tool_user_message(blocks, messages);
}

fn append_mixed_user_and_tool_result_messages(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut user_blocks = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("tool_result") => {
                let content = stringify_tool_result_content(block.get("content"));
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or_default(),
                    "content": content,
                }));
            }
            Some("text") | Some("image") => user_blocks.push(block.clone()),
            _ => {}
        }
    }

    append_non_tool_user_message(&user_blocks, messages);
}

fn append_non_tool_user_message(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut rich_parts: Vec<Value> = Vec::new();
    let mut has_images = false;

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if has_images {
                    rich_parts.push(json!({ "type": "text", "text": text }));
                } else {
                    text_parts.push(text);
                }
            }
            Some("image") => {
                if !has_images {
                    for text in text_parts.drain(..) {
                        rich_parts.push(json!({ "type": "text", "text": text }));
                    }
                }
                has_images = true;
                let source = block.get("source").unwrap_or(&Value::Null);
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                let data = source
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                rich_parts.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{data}")
                    }
                }));
            }
            _ => {}
        }
    }

    push_pending_user_message(&mut text_parts, &mut rich_parts, has_images, messages);
}

fn push_pending_user_message(
    text_parts: &mut Vec<String>,
    rich_parts: &mut Vec<Value>,
    has_images: bool,
    messages: &mut Vec<Value>,
) {
    if has_images {
        if !rich_parts.is_empty() {
            messages.push(json!({
                "role": "user",
                "content": Value::Array(std::mem::take(rich_parts)),
            }));
        }
        text_parts.clear();
        return;
    }

    let text = text_parts
        .iter()
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !text.is_empty() {
        messages.push(json!({
            "role": "user",
            "content": text,
        }));
    }
    text_parts.clear();
    rich_parts.clear();
}

fn build_assistant_message(blocks: &[Value]) -> Value {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("tool_use") => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": block.get("id").and_then(Value::as_str).unwrap_or_default(),
                    "type": "function",
                    "function": {
                        "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": input.to_string(),
                    }
                }));
            }
            _ => {}
        }
    }

    let mut message = json!({
        "role": "assistant",
        "content": if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n\n"))
        },
    });

    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    message
}

fn openai_response_to_anthropic(response: &Value, requested_model: &str) -> Result<Value, String> {
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| "OpenAI response is missing choices".to_string())?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAI response is missing message".to_string())?;

    let mut content = Vec::new();
    if let Some(text) = extract_openai_text(message.get("content")) {
        if !text.trim().is_empty() {
            content.push(json!({
                "type": "text",
                "text": text,
            }));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let arguments = tool_call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let parsed_input = serde_json::from_str::<Value>(arguments)
                .unwrap_or_else(|_| Value::String(arguments.to_string()));
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": tool_call.pointer("/function/name").and_then(Value::as_str).unwrap_or_default(),
                "input": parsed_input,
            }));
        }
    }

    let stop_reason = match choice.get("finish_reason").and_then(Value::as_str) {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        Some("stop") => "end_turn",
        Some("content_filter") => "end_turn",
        _ => "end_turn",
    };

    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64);

    Ok(json!({
        "id": response.get("id").and_then(Value::as_str).unwrap_or_default(),
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": Value::Null,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
    }))
}

fn extract_openai_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            part.pointer("/text/value")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn message_content_blocks(content: Option<&Value>) -> Result<Vec<Value>, String> {
    match content {
        Some(Value::Array(blocks)) => Ok(blocks.clone()),
        Some(Value::String(text)) => Ok(vec![json!({
            "type": "text",
            "text": text,
        })]),
        Some(_) => Err("Anthropic message content must be a string or content array".to_string()),
        None => Err("Anthropic message is missing content".to_string()),
    }
}

fn flatten_anthropic_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.trim().to_string(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(extract_text_from_block)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn extract_text_from_block(block: &Value) -> Option<String> {
    block
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            block
                .pointer("/text/value")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn flatten_text_blocks(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(extract_text_from_block)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn stringify_tool_result_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(_)) => {
            let text = flatten_anthropic_content_text(content);
            if text.is_empty() {
                serde_json::to_string(content.unwrap_or(&Value::Null)).unwrap_or_default()
            } else {
                text
            }
        }
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        extract_last_user_text, extract_response_text, request_to_upstream, response_from_upstream,
        validate_request,
    };
    use crate::providers::Provider;
    use serde_json::{json, Value};

    #[test]
    fn translates_anthropic_request_to_openai_tools_format() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Find a repo"}]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_repo",
                        "input": {"query": "aura"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "done"
                    }]
                }
            ],
            "tools": [{
                "name": "search_repo",
                "description": "Search repositories",
                "input_schema": {"type": "object"}
            }],
            "tool_choice": {"type": "tool", "name": "search_repo"},
            "max_tokens": 2048
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["model"], "gpt-4.1");
        assert_eq!(translated["messages"][0]["role"], "system");
        assert_eq!(translated["messages"][1]["role"], "user");
        assert_eq!(
            translated["messages"][2]["tool_calls"][0]["function"]["name"],
            "search_repo"
        );
        assert_eq!(translated["messages"][3]["role"], "tool");
        assert_eq!(translated["tool_choice"]["function"]["name"], "search_repo");
        assert_eq!(translated["max_completion_tokens"], 2048);
    }

    #[test]
    fn translates_mixed_tool_results_before_follow_up_user_content() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_repo",
                        "input": {"query": "aura"}
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the screenshot."},
                        {
                            "type": "image",
                            "source": {
                                "media_type": "image/png",
                                "data": "ZmFrZQ=="
                            }
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "repo found"}]
                        },
                        {"type": "text", "text": "Please summarize it."}
                    ]
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");
        let messages = translated["messages"].as_array().expect("messages array");

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "toolu_1");
        assert_eq!(messages[1]["content"], "repo found");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "text");
        assert_eq!(messages[2]["content"][0]["text"], "Here is the screenshot.");
        assert_eq!(messages[2]["content"][1]["type"], "image_url");
        assert_eq!(messages[2]["content"][2]["type"], "text");
        assert_eq!(messages[2]["content"][2]["text"], "Please summarize it.");
    }

    #[test]
    fn translates_openai_response_to_anthropic_blocks() {
        let response = json!({
            "id": "chatcmpl_123",
            "model": "gpt-4.1",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Let me check that.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_repo",
                            "arguments": "{\"query\":\"aura\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7
            }
        });

        let translated = response_from_upstream(Provider::OpenAi, "aura-gpt-4.1", &response)
            .expect("translation");

        assert_eq!(translated["model"], "aura-gpt-4.1");
        assert_eq!(translated["stop_reason"], "tool_use");
        assert_eq!(translated["content"][0]["type"], "text");
        assert_eq!(translated["content"][1]["type"], "tool_use");
        assert_eq!(translated["usage"]["input_tokens"], 12);
        assert_eq!(translated["usage"]["output_tokens"], 7);
    }

    #[test]
    fn rejects_openai_requests_with_anthropic_thinking() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }],
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        });

        let error = validate_request(Provider::OpenAi, &request).expect_err("request should fail");
        assert!(error.contains("do not support Anthropic thinking"));
    }

    #[test]
    fn rejects_openai_requests_with_unsupported_message_blocks() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [{
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "secret"}]
            }]
        });

        let error = validate_request(Provider::OpenAi, &request).expect_err("request should fail");
        assert!(error.contains("do not support `thinking` blocks"));
    }

    #[test]
    fn rejects_fireworks_requests_with_store_flag() {
        let request = json!({
            "model": "aura-deepseek-v3-2",
            "store": true,
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        });

        let error =
            validate_request(Provider::Fireworks, &request).expect_err("request should fail");
        assert!(error.contains("do not allow request-level persistence controls"));
    }

    #[test]
    fn rejects_fireworks_requests_with_response_state_replay() {
        let request = json!({
            "model": "aura-deepseek-v3-2",
            "response_id": "resp_123",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        });

        let error =
            validate_request(Provider::Fireworks, &request).expect_err("request should fail");
        assert!(error.contains("stateless inference for privacy"));
    }

    #[test]
    fn translates_string_shorthand_messages_for_openai_requests() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello from shorthand"
                },
                {
                    "role": "assistant",
                    "content": "Partial answer"
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["messages"][0]["role"], "user");
        assert_eq!(translated["messages"][0]["content"], "Hello from shorthand");
        assert_eq!(translated["messages"][1]["role"], "assistant");
        assert_eq!(translated["messages"][1]["content"], "Partial answer");
    }

    #[test]
    fn forwards_streaming_and_high_value_compat_fields_for_openai_requests() {
        let request = json!({
            "model": "aura-gpt-4.1",
            "stream": true,
            "top_p": 0.9,
            "stop_sequences": ["DONE", "STOP"],
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        });

        let translated =
            request_to_upstream(Provider::OpenAi, "gpt-4.1", &request).expect("translation");

        assert_eq!(translated["stream"], Value::Bool(true));
        assert_eq!(
            translated["stream_options"]["include_usage"],
            Value::Bool(true)
        );
        assert_eq!(translated["top_p"], json!(0.9));
        assert_eq!(translated["stop"], json!(["DONE", "STOP"]));
    }

    #[test]
    fn extracts_last_user_text_from_string_and_block_content() {
        let request = json!({
            "messages": [
                {
                    "role": "user",
                    "content": "Earlier"
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Reply"}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Latest"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                        {"type": "text", "text": "Question"}
                    ]
                }
            ]
        });

        assert_eq!(extract_last_user_text(&request), "Latest\n\nQuestion");
    }

    #[test]
    fn extracts_response_text_from_text_blocks() {
        let response = json!({
            "content": [
                {"type": "text", "text": "First"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "aura"}},
                {"type": "text", "text": "Second"}
            ]
        });

        assert_eq!(extract_response_text(&response), "First\n\nSecond");
    }
}
