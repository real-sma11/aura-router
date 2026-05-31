//! Provider resolution — maps model names to LLM providers.

use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    Fireworks,
    DeepSeek,
    Google,
}

impl Provider {
    /// Provider name string for billing/stats.
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Fireworks => "fireworks",
            Provider::DeepSeek => "deepseek",
            Provider::Google => "google",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel<'a> {
    pub requested_model: &'a str,
    pub upstream_model: &'a str,
    pub provider: Provider,
}

fn aura_model_alias(model: &str) -> Option<ResolvedModel<'_>> {
    match model {
        "aura-claude-opus-4-8" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-8",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-4-7" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-7",
            provider: Provider::Anthropic,
        }),
        "aura-claude-opus-4-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-opus-4-6",
            provider: Provider::Anthropic,
        }),
        "aura-claude-sonnet-4-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-sonnet-4-6",
            provider: Provider::Anthropic,
        }),
        "aura-claude-haiku-4-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "claude-haiku-4-5",
            provider: Provider::Anthropic,
        }),
        "aura-gpt-5-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.5",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4-mini" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4-mini",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-5-4-nano" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-5.4-nano",
            provider: Provider::OpenAi,
        }),
        "aura-gpt-4.1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gpt-4.1",
            provider: Provider::OpenAi,
        }),
        "aura-o3" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "o3",
            provider: Provider::OpenAi,
        }),
        "aura-o4-mini" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "o4-mini",
            provider: Provider::OpenAi,
        }),
        // DeepSeek V4 models are served via Fireworks (which hosts them
        // verbatim) rather than DeepSeek's first-party API, so they reuse the
        // already-provisioned FIREWORKS_API_KEY. Provider::DeepSeek remains for
        // raw first-party passthrough names (see infer_provider).
        "aura-deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-pro",
            provider: Provider::Fireworks,
        }),
        "aura-deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-flash",
            provider: Provider::Fireworks,
        }),
        "deepseek/deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-pro",
            provider: Provider::Fireworks,
        }),
        "deepseek/deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/deepseek-v4-flash",
            provider: Provider::Fireworks,
        }),
        "aura-kimi-k2-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p5",
            provider: Provider::Fireworks,
        }),
        "aura-kimi-k2-6" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p6",
            provider: Provider::Fireworks,
        }),
        "aura-oss-120b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/gpt-oss-120b",
            provider: Provider::Fireworks,
        }),
        "aura-qwen2-5-coder-7b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/qwen2p5-coder-7b",
            provider: Provider::Fireworks,
        }),
        "aura-minimax-m2-7" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/minimax-m2p7",
            provider: Provider::Fireworks,
        }),
        "aura-glm-5-1" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/glm-5p1",
            provider: Provider::Fireworks,
        }),
        "aura-qwen3-6-plus" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/qwen3p6-plus",
            provider: Provider::Fireworks,
        }),
        "aura-gemma-4-31b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/gemma-4-31b-it",
            provider: Provider::Fireworks,
        }),
        "aura-gemma-4-26b-a4b" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/gemma-4-26b-a4b-it",
            provider: Provider::Fireworks,
        }),
        // Google Gemini chat models route directly through the Google
        // Generative Language API (`generativelanguage.googleapis.com`)
        // using the platform GOOGLE_API_KEY. Pro models map to their
        // current preview string; stable Flash/Flash-Lite tiers map to the
        // bare stable name.
        "aura-gemini-3-1-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.1-pro-preview",
            provider: Provider::Google,
        }),
        "aura-gemini-3-5-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.5-flash",
            provider: Provider::Google,
        }),
        "aura-gemini-3-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3-flash-preview",
            provider: Provider::Google,
        }),
        "aura-gemini-3-1-flash-lite" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-3.1-flash-lite",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-pro",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-flash",
            provider: Provider::Google,
        }),
        "aura-gemini-2-5-flash-lite" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "gemini-2.5-flash-lite",
            provider: Provider::Google,
        }),
        _ => None,
    }
}

fn infer_provider(model: &str) -> Option<Provider> {
    if model.starts_with("claude") {
        Some(Provider::Anthropic)
    } else if model.starts_with("gpt")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("codex")
    {
        Some(Provider::OpenAi)
    } else if model.starts_with("deepseek-v4")
        || model == "deepseek-chat"
        || model == "deepseek-reasoner"
    {
        Some(Provider::DeepSeek)
    } else if model.starts_with("gemini") {
        Some(Provider::Google)
    } else {
        None
    }
}

/// Resolve an Aura or upstream model name into its upstream provider/model pair.
pub fn resolve_model(model: &str) -> Option<ResolvedModel<'_>> {
    if let Some(alias) = aura_model_alias(model) {
        return Some(alias);
    }

    infer_provider(model).map(|provider| ResolvedModel {
        requested_model: model,
        upstream_model: model,
        provider,
    })
}

/// Resolve a model name to its provider.
pub fn resolve_provider(model: &str) -> Option<Provider> {
    resolve_model(model).map(|resolved| resolved.provider)
}

/// Which OpenAI HTTP surface a request should use.
///
/// OpenAI's `/v1/chat/completions` rejects requests that combine function
/// `tools` with `reasoning_effort` for the gpt-5 family ("Please use
/// /v1/responses instead"). The `/v1/responses` API supports tools +
/// reasoning together and is OpenAI's go-forward surface for tool use, so
/// any tool-bearing OpenAI request is routed through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiApi {
    ChatCompletions,
    Responses,
}

/// Decide which OpenAI surface an inbound `/v1/messages` request maps to.
///
/// Returns [`OpenAiApi::Responses`] only for OpenAI requests that carry a
/// non-empty `tools` array; everything else (including all non-OpenAI
/// providers, for which the value is ignored) reports
/// [`OpenAiApi::ChatCompletions`].
pub fn openai_api_for_request(provider: Provider, request: &Value) -> OpenAiApi {
    if provider != Provider::OpenAi {
        return OpenAiApi::ChatCompletions;
    }
    let has_tools = request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if has_tools {
        OpenAiApi::Responses
    } else {
        OpenAiApi::ChatCompletions
    }
}

/// Get the base URL for a provider's messages endpoint.
///
/// For OpenAI this returns the chat/completions surface; callers that may
/// need the Responses API should use [`openai_endpoint_url`] with the
/// [`OpenAiApi`] resolved from the request.
pub fn provider_url(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages",
        Provider::OpenAi => "https://api.openai.com/v1/chat/completions",
        // Intentionally use the stateless chat completions path for Aura's OSS lane.
        // Aura Router centrally avoids Fireworks surfaces that can retain conversation state.
        Provider::Fireworks => "https://api.fireworks.ai/inference/v1/chat/completions",
        Provider::DeepSeek => "https://api.deepseek.com/chat/completions",
        // Google encodes the model + streaming mode in the path, so callers
        // must use `google_endpoint_url` instead. This base is returned only
        // to keep the match exhaustive.
        Provider::Google => "https://generativelanguage.googleapis.com/v1beta",
    }
}

/// Google Generative Language endpoint for a Gemini chat request.
///
/// Unlike the other providers, Gemini puts the upstream model and the
/// streaming mode in the URL path (`:generateContent` vs
/// `:streamGenerateContent?alt=sse`) rather than the request body.
pub fn google_endpoint_url(upstream_model: &str, streaming: bool) -> String {
    let method = if streaming {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    format!("https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:{method}")
}

/// Get the OpenAI endpoint URL for the given API surface.
pub fn openai_endpoint_url(api: OpenAiApi) -> &'static str {
    match api {
        OpenAiApi::ChatCompletions => "https://api.openai.com/v1/chat/completions",
        OpenAiApi::Responses => "https://api.openai.com/v1/responses",
    }
}

/// Get the maximum context window size for a model (in tokens).
pub fn max_context_tokens(model: &str) -> u64 {
    let resolved_model = resolve_model(model)
        .map(|resolved| resolved.upstream_model)
        .unwrap_or(model);
    match resolved_model {
        // Anthropic
        m if m.starts_with("claude-opus-4") => 1_000_000,
        m if m.starts_with("claude-sonnet-4") => 1_000_000,
        m if m.starts_with("claude-haiku-4") => 200_000,
        m if m.starts_with("claude-3-5") => 200_000,
        m if m.starts_with("claude-3") => 200_000,
        m if m.starts_with("claude") => 200_000,
        // OpenAI
        "gpt-5.5" => 1_000_000,
        "gpt-5.4" => 1_050_000,
        "gpt-5.4-mini" => 400_000,
        "gpt-5.4-nano" => 400_000,
        m if m.starts_with("gpt-4o") => 128_000,
        m if m.starts_with("gpt-4-turbo") => 128_000,
        m if m.starts_with("gpt-4") => 8_192,
        m if m.starts_with("gpt-3.5") => 16_385,
        m if m.starts_with("o1") => 200_000,
        m if m.starts_with("o3") => 200_000,
        m if m.starts_with("o4") => 200_000,
        m if m.starts_with("codex") => 200_000,
        // Fireworks OSS
        "accounts/fireworks/models/kimi-k2p5" => 262_144,
        "accounts/fireworks/models/kimi-k2p6" => 262_144,
        "accounts/fireworks/models/gpt-oss-120b" => 131_072,
        "accounts/fireworks/models/qwen2p5-coder-7b" => 32_768,
        "accounts/fireworks/models/minimax-m2p7" => 196_608,
        "accounts/fireworks/models/glm-5p1" => 202_752,
        "accounts/fireworks/models/qwen3p6-plus" => 262_144,
        "accounts/fireworks/models/gemma-4-31b-it" => 262_144,
        "accounts/fireworks/models/gemma-4-26b-a4b-it" => 262_144,
        // DeepSeek V4 via Fireworks
        "accounts/fireworks/models/deepseek-v4-pro"
        | "accounts/fireworks/models/deepseek-v4-flash" => 1_000_000,
        // DeepSeek V4 direct API
        "deepseek-v4-pro" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            1_000_000
        }
        // Google Gemini — the 2.5 and 3 families all expose a 1M token window.
        m if m.starts_with("gemini") => 1_000_000,
        _ => 200_000, // safe default
    }
}

/// Build provider-specific headers for the upstream request.
///
/// Returns None if the API key contains invalid header characters.
pub fn provider_headers(provider: &Provider, api_key: &str) -> Option<HeaderMap> {
    let mut headers = HeaderMap::new();
    match provider {
        Provider::Anthropic => {
            headers.insert("x-api-key", HeaderValue::from_str(api_key).ok()?);
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_static("prompt-caching-2024-07-31"),
            );
        }
        Provider::OpenAi | Provider::DeepSeek => {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).ok()?,
            );
        }
        Provider::Fireworks => {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {api_key}")).ok()?,
            );
        }
        Provider::Google => {
            // The Generative Language API authenticates with a bare API key
            // header rather than a Bearer token.
            headers.insert("x-goog-api-key", HeaderValue::from_str(api_key).ok()?);
        }
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Some(headers)
}

#[cfg(test)]
mod tests {
    use super::{
        openai_api_for_request, openai_endpoint_url, resolve_model, resolve_provider, OpenAiApi,
        Provider,
    };
    use serde_json::json;

    #[test]
    fn openai_tool_requests_route_to_responses_api() {
        let request = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "search", "input_schema": {} }],
        });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &request),
            OpenAiApi::Responses
        );
        assert_eq!(
            openai_endpoint_url(openai_api_for_request(Provider::OpenAi, &request)),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn openai_requests_without_tools_use_chat_completions() {
        let request = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &request),
            OpenAiApi::ChatCompletions
        );
        let empty_tools = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [],
        });
        assert_eq!(
            openai_api_for_request(Provider::OpenAi, &empty_tools),
            OpenAiApi::ChatCompletions
        );
    }

    #[test]
    fn non_openai_providers_never_select_responses_api() {
        let request = json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "name": "search", "input_schema": {} }],
        });
        for provider in [Provider::Anthropic, Provider::Fireworks, Provider::DeepSeek] {
            assert_eq!(
                openai_api_for_request(provider, &request),
                OpenAiApi::ChatCompletions
            );
        }
    }

    #[test]
    fn resolves_aura_aliases_to_upstream_models() {
        let resolved = resolve_model("aura-gpt-5-5").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-gpt-5-5");
        assert_eq!(resolved.upstream_model, "gpt-5.5");
        assert_eq!(resolved.provider, Provider::OpenAi);

        let resolved = resolve_model("aura-gpt-5-4-mini").expect("model alias should resolve");
        assert_eq!(resolved.requested_model, "aura-gpt-5-4-mini");
        assert_eq!(resolved.upstream_model, "gpt-5.4-mini");
        assert_eq!(resolved.provider, Provider::OpenAi);
    }

    #[test]
    fn preserves_upstream_model_names() {
        let resolved = resolve_model("claude-sonnet-4-6").expect("provider should resolve");
        assert_eq!(resolved.requested_model, "claude-sonnet-4-6");
        assert_eq!(resolved.upstream_model, "claude-sonnet-4-6");
        assert_eq!(resolved.provider, Provider::Anthropic);
    }

    #[test]
    fn resolve_provider_understands_aura_aliases() {
        assert_eq!(resolve_provider("aura-gpt-5-5"), Some(Provider::OpenAi));
        assert_eq!(resolve_provider("aura-gpt-5-4"), Some(Provider::OpenAi));
        assert_eq!(
            resolve_provider("aura-kimi-k2-5"),
            Some(Provider::Fireworks)
        );
        assert_eq!(
            resolve_provider("aura-kimi-k2-6"),
            Some(Provider::Fireworks)
        );
        assert_eq!(
            resolve_provider("aura-deepseek-v4-flash"),
            Some(Provider::Fireworks)
        );
    }

    #[test]
    fn resolves_deepseek_v4_models_via_fireworks() {
        // Picker-facing DeepSeek aliases route through Fireworks (which hosts
        // the V4 models verbatim) and reuse FIREWORKS_API_KEY.
        let resolved = resolve_model("aura-deepseek-v4-pro").expect("aura alias");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/deepseek-v4-pro"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);

        let resolved = resolve_model("deepseek/deepseek-v4-flash").expect("provider alias");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/deepseek-v4-flash"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);

        // Raw first-party names still resolve to the direct DeepSeek API.
        let resolved = resolve_model("deepseek-chat").expect("compat alias");
        assert_eq!(resolved.upstream_model, "deepseek-chat");
        assert_eq!(resolved.provider, Provider::DeepSeek);
    }

    #[test]
    fn resolves_kimi_k2_6_to_fireworks() {
        let resolved = resolve_model("aura-kimi-k2-6").expect("aura alias should resolve");
        assert_eq!(
            resolved.upstream_model,
            "accounts/fireworks/models/kimi-k2p6"
        );
        assert_eq!(resolved.provider, Provider::Fireworks);
    }

    #[test]
    fn resolves_new_fireworks_models() {
        for (alias, upstream) in [
            ("aura-minimax-m2-7", "accounts/fireworks/models/minimax-m2p7"),
            ("aura-glm-5-1", "accounts/fireworks/models/glm-5p1"),
            ("aura-qwen3-6-plus", "accounts/fireworks/models/qwen3p6-plus"),
            ("aura-gemma-4-31b", "accounts/fireworks/models/gemma-4-31b-it"),
            (
                "aura-gemma-4-26b-a4b",
                "accounts/fireworks/models/gemma-4-26b-a4b-it",
            ),
        ] {
            let resolved = resolve_model(alias).expect("aura alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Fireworks);
        }
    }

    #[test]
    fn resolves_gpt_5_5_api_model() {
        let resolved = resolve_model("gpt-5.5").expect("api model should resolve");
        assert_eq!(resolved.requested_model, "gpt-5.5");
        assert_eq!(resolved.upstream_model, "gpt-5.5");
        assert_eq!(resolved.provider, Provider::OpenAi);
    }

    #[test]
    fn resolves_gemini_aliases_to_google() {
        for (alias, upstream) in [
            ("aura-gemini-3-1-pro", "gemini-3.1-pro-preview"),
            ("aura-gemini-3-5-flash", "gemini-3.5-flash"),
            ("aura-gemini-3-flash", "gemini-3-flash-preview"),
            ("aura-gemini-3-1-flash-lite", "gemini-3.1-flash-lite"),
            ("aura-gemini-2-5-pro", "gemini-2.5-pro"),
            ("aura-gemini-2-5-flash", "gemini-2.5-flash"),
            ("aura-gemini-2-5-flash-lite", "gemini-2.5-flash-lite"),
        ] {
            let resolved = resolve_model(alias).expect("aura gemini alias should resolve");
            assert_eq!(resolved.upstream_model, upstream);
            assert_eq!(resolved.provider, Provider::Google);
        }
    }

    #[test]
    fn infers_google_provider_for_raw_gemini_names() {
        let resolved = resolve_model("gemini-2.5-flash").expect("api model should resolve");
        assert_eq!(resolved.upstream_model, "gemini-2.5-flash");
        assert_eq!(resolved.provider, Provider::Google);
        assert_eq!(super::max_context_tokens("aura-gemini-2-5-pro"), 1_000_000);
    }

    #[test]
    fn google_endpoint_url_encodes_model_and_streaming_mode() {
        assert_eq!(
            super::google_endpoint_url("gemini-2.5-pro", false),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        );
        assert_eq!(
            super::google_endpoint_url("gemini-2.5-pro", true),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }
}
