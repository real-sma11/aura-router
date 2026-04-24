//! Provider resolution — maps model names to LLM providers.

use reqwest::header::{HeaderMap, HeaderValue};

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAi,
    Fireworks,
    DeepSeek,
}

impl Provider {
    /// Provider name string for billing/stats.
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAi => "openai",
            Provider::Fireworks => "fireworks",
            Provider::DeepSeek => "deepseek",
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
        "aura-deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "deepseek-v4-pro",
            provider: Provider::DeepSeek,
        }),
        "aura-deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "deepseek-v4-flash",
            provider: Provider::DeepSeek,
        }),
        "deepseek/deepseek-v4-pro" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "deepseek-v4-pro",
            provider: Provider::DeepSeek,
        }),
        "deepseek/deepseek-v4-flash" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "deepseek-v4-flash",
            provider: Provider::DeepSeek,
        }),
        "aura-kimi-k2-5" => Some(ResolvedModel {
            requested_model: model,
            upstream_model: "accounts/fireworks/models/kimi-k2p5",
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
        _ => None,
    }
}

fn infer_provider(model: &str) -> Option<Provider> {
    if model == "gpt-5.5" {
        None
    } else if model.starts_with("claude") {
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

/// Get the base URL for a provider's messages endpoint.
pub fn provider_url(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "https://api.anthropic.com/v1/messages",
        Provider::OpenAi => "https://api.openai.com/v1/chat/completions",
        // Intentionally use the stateless chat completions path for Aura's OSS lane.
        // Aura Router centrally avoids Fireworks surfaces that can retain conversation state.
        Provider::Fireworks => "https://api.fireworks.ai/inference/v1/chat/completions",
        Provider::DeepSeek => "https://api.deepseek.com/chat/completions",
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
        "accounts/fireworks/models/gpt-oss-120b" => 131_072,
        "accounts/fireworks/models/qwen2p5-coder-7b" => 32_768,
        // DeepSeek V4 direct API
        "deepseek-v4-pro" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            1_000_000
        }
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
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Some(headers)
}

#[cfg(test)]
mod tests {
    use super::{resolve_model, resolve_provider, Provider};

    #[test]
    fn resolves_aura_aliases_to_upstream_models() {
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
        assert_eq!(resolve_provider("aura-gpt-5-4"), Some(Provider::OpenAi));
        assert_eq!(
            resolve_provider("aura-kimi-k2-5"),
            Some(Provider::Fireworks)
        );
        assert_eq!(
            resolve_provider("aura-deepseek-v4-flash"),
            Some(Provider::DeepSeek)
        );
    }

    #[test]
    fn resolves_deepseek_v4_models_to_direct_api() {
        let resolved = resolve_model("aura-deepseek-v4-pro").expect("aura alias");
        assert_eq!(resolved.upstream_model, "deepseek-v4-pro");
        assert_eq!(resolved.provider, Provider::DeepSeek);

        let resolved = resolve_model("deepseek/deepseek-v4-flash").expect("provider alias");
        assert_eq!(resolved.upstream_model, "deepseek-v4-flash");
        assert_eq!(resolved.provider, Provider::DeepSeek);

        let resolved = resolve_model("deepseek-chat").expect("compat alias");
        assert_eq!(resolved.upstream_model, "deepseek-chat");
        assert_eq!(resolved.provider, Provider::DeepSeek);
    }

    #[test]
    fn does_not_resolve_undocumented_gpt_5_5() {
        assert_eq!(resolve_model("aura-gpt-5-5"), None);
        assert_eq!(resolve_model("gpt-5.5"), None);
        assert_eq!(resolve_provider("gpt-5.5"), None);
    }
}
