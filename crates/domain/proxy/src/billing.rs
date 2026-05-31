//! z-billing client for credit checks and usage reporting.

use serde::{Deserialize, Serialize};

use aura_router_core::AppError;

/// Response from POST /v1/usage/check.
#[derive(Debug, Deserialize)]
pub struct CheckBalanceResponse {
    pub sufficient: bool,
    pub balance_cents: i64,
    pub required_cents: i64,
}

/// Response from POST /v1/usage.
#[derive(Debug, Deserialize)]
pub struct UsageResponse {
    pub success: bool,
    pub balance_cents: i64,
    pub cost_cents: i64,
    pub transaction_id: String,
}

/// Usage metric for LLM tokens.
#[derive(Debug, Serialize)]
pub struct LlmMetric {
    #[serde(rename = "type")]
    pub metric_type: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

const DEFAULT_LLM_MARKUP_MULTIPLIER: f64 = 1.20;

#[derive(Debug, Clone, Copy)]
struct CacheAwareRates {
    /// Rate for uncached input tokens (the prompt portion that is neither
    /// served from cache nor written to cache).
    new_input_cents_per_million: f64,
    /// Rate for tokens being written to cache. For providers that bill
    /// cache writes at the same rate as base input (OpenAI, DeepSeek,
    /// Fireworks), this equals `new_input_cents_per_million`. For
    /// providers that charge a premium for cache writes (Anthropic 5m
    /// ephemeral cache = 1.25× base input), this is higher.
    cache_write_input_cents_per_million: f64,
    /// Rate for tokens served from cache (always cheaper than base input).
    cache_read_input_cents_per_million: f64,
    output_cents_per_million: f64,
    /// Whether the provider's `input_tokens` field reports only the new
    /// (uncached) input portion (Anthropic), or the total request input
    /// including tokens served from cache (OpenAI/DeepSeek/Fireworks).
    input_tokens_is_new_only: bool,
}

/// Calculate a cache-aware cost override for providers whose raw usage has
/// more detail than z-billing's generic input/output metric.
pub fn cache_aware_cost_cents(
    provider: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
) -> Option<i64> {
    if cache_creation_input_tokens == 0 && cache_read_input_tokens == 0 {
        return None;
    }
    let rates = cache_aware_rates(provider, model)?;

    let new_input_tokens = if rates.input_tokens_is_new_only {
        input_tokens
    } else {
        input_tokens
            .saturating_sub(cache_creation_input_tokens.saturating_add(cache_read_input_tokens))
    };

    let total_cents = ((new_input_tokens as f64 * rates.new_input_cents_per_million)
        + (cache_creation_input_tokens as f64 * rates.cache_write_input_cents_per_million)
        + (cache_read_input_tokens as f64 * rates.cache_read_input_cents_per_million)
        + (output_tokens as f64 * rates.output_cents_per_million))
        * DEFAULT_LLM_MARKUP_MULTIPLIER
        / 1_000_000.0;

    let rounded = total_cents.round() as i64;
    if rounded == 0 && (input_tokens > 0 || output_tokens > 0) {
        Some(1)
    } else {
        Some(rounded)
    }
}

fn cache_aware_rates(provider: &str, model: &str) -> Option<CacheAwareRates> {
    match provider {
        "anthropic" => anthropic_rates(model),
        "deepseek" => deepseek_rates(model),
        "openai" => openai_rates(model),
        "fireworks" => fireworks_rates(model),
        _ => None,
    }
}

fn anthropic_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("anthropic/").unwrap_or(model);

    // Anthropic returns `input_tokens` as the new (uncached) portion only;
    // cache creation and cache read tokens are reported separately.
    // 5-minute ephemeral cache: write = 1.25× base input, read = 0.10× base input.
    match model {
        "claude-opus-4-6"
        | "claude-opus-4-7"
        | "claude-opus-4-8"
        | "aura-claude-opus-4-6"
        | "aura-claude-opus-4-7"
        | "aura-claude-opus-4-8" => Some(CacheAwareRates {
            new_input_cents_per_million: 500.0,
            cache_write_input_cents_per_million: 625.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 2500.0,
            input_tokens_is_new_only: true,
        }),
        "claude-sonnet-4-6" | "aura-claude-sonnet-4-6" => Some(CacheAwareRates {
            new_input_cents_per_million: 300.0,
            cache_write_input_cents_per_million: 375.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 1500.0,
            input_tokens_is_new_only: true,
        }),
        "claude-haiku-4-5" | "claude-haiku-4-5-20251001" | "aura-claude-haiku-4-5" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 100.0,
                cache_write_input_cents_per_million: 125.0,
                cache_read_input_cents_per_million: 10.0,
                output_cents_per_million: 500.0,
                input_tokens_is_new_only: true,
            })
        }
        _ => None,
    }
}

fn deepseek_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("deepseek/").unwrap_or(model);

    match model {
        "aura-deepseek-v4-pro" | "deepseek-v4-pro" => Some(CacheAwareRates {
            new_input_cents_per_million: 174.0,
            cache_write_input_cents_per_million: 174.0,
            cache_read_input_cents_per_million: 14.5,
            output_cents_per_million: 348.0,
            input_tokens_is_new_only: false,
        }),
        "aura-deepseek-v4-flash" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 14.0,
                cache_write_input_cents_per_million: 14.0,
                cache_read_input_cents_per_million: 2.8,
                output_cents_per_million: 28.0,
                input_tokens_is_new_only: false,
            })
        }
        _ => None,
    }
}

fn openai_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("openai/").unwrap_or(model);

    match model {
        "aura-gpt-5-5" | "gpt-5.5" => Some(CacheAwareRates {
            new_input_cents_per_million: 500.0,
            cache_write_input_cents_per_million: 500.0,
            cache_read_input_cents_per_million: 50.0,
            output_cents_per_million: 3000.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4" | "gpt-5.4" => Some(CacheAwareRates {
            new_input_cents_per_million: 250.0,
            cache_write_input_cents_per_million: 250.0,
            cache_read_input_cents_per_million: 25.0,
            output_cents_per_million: 1500.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4-mini" | "gpt-5.4-mini" => Some(CacheAwareRates {
            new_input_cents_per_million: 75.0,
            cache_write_input_cents_per_million: 75.0,
            cache_read_input_cents_per_million: 7.5,
            output_cents_per_million: 450.0,
            input_tokens_is_new_only: false,
        }),
        "aura-gpt-5-4-nano" | "gpt-5.4-nano" => Some(CacheAwareRates {
            new_input_cents_per_million: 20.0,
            cache_write_input_cents_per_million: 20.0,
            cache_read_input_cents_per_million: 2.0,
            output_cents_per_million: 125.0,
            input_tokens_is_new_only: false,
        }),
        _ => None,
    }
}

fn fireworks_rates(model: &str) -> Option<CacheAwareRates> {
    let model = model.strip_prefix("fireworks/").unwrap_or(model);

    // Fireworks normalizes through the OpenAI-shaped path, so `input_tokens`
    // is the total prompt size. Fireworks discounts cache reads at 50% off
    // base input and does not charge a premium for cache writes.
    match model {
        "aura-kimi-k2-5" | "accounts/fireworks/models/kimi-k2p5" => Some(CacheAwareRates {
            new_input_cents_per_million: 60.0,
            cache_write_input_cents_per_million: 60.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 300.0,
            input_tokens_is_new_only: false,
        }),
        "accounts/fireworks/models/kimi-k2p5-turbo"
        | "accounts/fireworks/routers/kimi-k2p5-turbo" => Some(CacheAwareRates {
            new_input_cents_per_million: 99.0,
            cache_write_input_cents_per_million: 99.0,
            cache_read_input_cents_per_million: 49.5,
            output_cents_per_million: 494.0,
            input_tokens_is_new_only: false,
        }),
        "aura-kimi-k2-6" | "accounts/fireworks/models/kimi-k2p6" => Some(CacheAwareRates {
            new_input_cents_per_million: 95.0,
            cache_write_input_cents_per_million: 95.0,
            cache_read_input_cents_per_million: 47.5,
            output_cents_per_million: 400.0,
            input_tokens_is_new_only: false,
        }),
        "accounts/fireworks/models/kimi-k2p6-turbo"
        | "accounts/fireworks/routers/kimi-k2p6-turbo" => Some(CacheAwareRates {
            new_input_cents_per_million: 200.0,
            cache_write_input_cents_per_million: 200.0,
            cache_read_input_cents_per_million: 100.0,
            output_cents_per_million: 800.0,
            input_tokens_is_new_only: false,
        }),
        "accounts/fireworks/models/kimi-k2-thinking"
        | "accounts/fireworks/models/kimi-k2-instruct-0905" => Some(CacheAwareRates {
            new_input_cents_per_million: 60.0,
            cache_write_input_cents_per_million: 60.0,
            cache_read_input_cents_per_million: 30.0,
            output_cents_per_million: 250.0,
            input_tokens_is_new_only: false,
        }),
        "aura-oss-120b" | "accounts/fireworks/models/gpt-oss-120b" => Some(CacheAwareRates {
            new_input_cents_per_million: 15.0,
            cache_write_input_cents_per_million: 15.0,
            cache_read_input_cents_per_million: 7.5,
            output_cents_per_million: 60.0,
            input_tokens_is_new_only: false,
        }),
        // DeepSeek V4 models are served via Fireworks, so they bill under the
        // "fireworks" provider. Base rates match DeepSeek's published pricing;
        // the standard markup is applied by `cache_aware_cost_cents`.
        "aura-deepseek-v4-pro" | "accounts/fireworks/models/deepseek-v4-pro" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 174.0,
                cache_write_input_cents_per_million: 174.0,
                cache_read_input_cents_per_million: 14.5,
                output_cents_per_million: 348.0,
                input_tokens_is_new_only: false,
            })
        }
        "aura-deepseek-v4-flash" | "accounts/fireworks/models/deepseek-v4-flash" => {
            Some(CacheAwareRates {
                new_input_cents_per_million: 14.0,
                cache_write_input_cents_per_million: 14.0,
                cache_read_input_cents_per_million: 2.8,
                output_cents_per_million: 28.0,
                input_tokens_is_new_only: false,
            })
        }
        _ => None,
    }
}

/// Pre-check whether a user has sufficient credits.
pub async fn check_credits(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    user_id: &str,
    required_cents: i64,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<CheckBalanceResponse, AppError> {
    let url = format!("{billing_url}/v1/usage/check");
    let mut request_body = serde_json::json!({
        "user_id": user_id,
        "required_cents": required_cents
    });
    if let Some(provider) = provider {
        request_body["provider"] = serde_json::Value::String(provider.to_string());
    }
    if let Some(model) = model {
        request_body["model"] = serde_json::Value::String(model.to_string());
    }

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::BillingError(format!(
            "z-billing check failed ({status}): {body}"
        )));
    }

    resp.json::<CheckBalanceResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

/// Report image generation usage and debit a flat cost.
pub async fn report_image_usage(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    event_id: &str,
    user_id: &str,
    provider: &str,
    model: &str,
    cost_cents: i64,
) -> Result<UsageResponse, AppError> {
    let url = format!("{billing_url}/v1/usage");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&serde_json::json!({
            "event_id": event_id,
            "user_id": user_id,
            "cost_cents": cost_cents,
            "metric": {
                "type": "llm_tokens",
                "provider": provider,
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0
            }
        }))
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "z-billing image usage report failed");
        return Err(AppError::BillingError(format!(
            "z-billing usage report failed ({status})"
        )));
    }

    resp.json::<UsageResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

/// Report LLM usage and debit credits.
pub async fn report_usage(
    client: &reqwest::Client,
    billing_url: &str,
    api_key: &str,
    event_id: &str,
    user_id: &str,
    provider: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_cents: Option<i64>,
) -> Result<UsageResponse, AppError> {
    let url = format!("{billing_url}/v1/usage");
    let mut body = serde_json::json!({
        "event_id": event_id,
        "user_id": user_id,
        "metric": {
            "type": "llm_tokens",
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    });
    if let Some(cost_cents) = cost_cents {
        body["cost_cents"] = serde_json::Value::from(cost_cents);
    }

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("x-service-name", "aura-router")
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing unreachable: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "z-billing usage report failed");
        return Err(AppError::BillingError(format!(
            "z-billing usage report failed ({status})"
        )));
    }

    resp.json::<UsageResponse>()
        .await
        .map_err(|e| AppError::BillingError(format!("z-billing response parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::cache_aware_cost_cents;

    #[test]
    fn deepseek_cache_aware_cost_uses_cache_buckets_and_markup() {
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "aura-deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            Some(226)
        );
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "deepseek/deepseek-v4-flash",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            Some(20)
        );
    }

    #[test]
    fn deepseek_cost_override_is_absent_without_cache_buckets() {
        assert_eq!(
            cache_aware_cost_cents(
                "deepseek",
                "aura-deepseek-v4-flash",
                1_000_000,
                500_000,
                0,
                0,
            ),
            None
        );
        // Unknown provider with cache buckets still returns None.
        assert_eq!(
            cache_aware_cost_cents("cohere", "command-r-plus", 1_000_000, 500_000, 0, 1_000_000,),
            None
        );
    }

    #[test]
    fn openai_cache_aware_cost_discounts_cached_tokens() {
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 1_000_000, 500_000, 0, 1_000_000,),
            Some(1860)
        );
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-5.4-mini", 1_000_000, 500_000, 0, 1_000_000,),
            Some(279)
        );
    }

    #[test]
    fn openai_cost_override_is_absent_without_cached_tokens_or_rates() {
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 1_000_000, 500_000, 0, 0),
            None
        );
        assert_eq!(
            cache_aware_cost_cents("openai", "gpt-4.1", 1_000_000, 500_000, 0, 1_000_000),
            None
        );
    }

    #[test]
    fn anthropic_opus_charges_cache_read_tokens() {
        // Mirrors a realistic Claude Code follow-up turn: tiny new input,
        // most of the prompt served from cache, modest output.
        // new: 1 × 500 = 500
        // read: 50_000 × 50 = 2_500_000
        // output: 595 × 2500 = 1_487_500
        // total: 3_988_000 / 1M = 3.988 × 1.20 = 4.7856 → 5
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 1, 595, 0, 50_000),
            Some(5)
        );
    }

    #[test]
    fn anthropic_sonnet_charges_cache_creation_at_premium() {
        // Cache-write turn with no new input outside the cache block.
        // write: 50_000 × 375 = 18_750_000
        // output: 600 × 1500 = 900_000
        // total: 19_650_000 / 1M = 19.65 × 1.20 = 23.58 → 24
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-sonnet-4-6", 0, 600, 50_000, 0),
            Some(24)
        );
    }

    #[test]
    fn anthropic_haiku_applies_minimum_credit_floor() {
        // Tiny call: rounded cost is < 1 cent but cache tokens are present,
        // so the 1-credit floor applies.
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-haiku-4-5-20251001", 10, 100, 0, 1_000),
            Some(1)
        );
    }

    #[test]
    fn anthropic_treats_input_tokens_as_new_only() {
        // For Anthropic, `input_tokens` already excludes cache_read tokens.
        // The function must NOT subtract cache_read from input_tokens.
        // new: 10 × 500 = 5_000
        // read: 100_000 × 50 = 5_000_000
        // total: 5_005_000 / 1M = 5.005 × 1.20 = 6.006 → 6
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 10, 0, 0, 100_000),
            Some(6)
        );
    }

    #[test]
    fn anthropic_cost_override_is_absent_without_cache_tokens() {
        // No cache fields → fall back to z-billing's token-only math.
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 100, 100, 0, 0),
            None
        );
    }

    #[test]
    fn anthropic_aura_aliases_resolve_to_same_rates() {
        // aura-* aliases must produce identical costs to canonical model names.
        let canonical = cache_aware_cost_cents("anthropic", "claude-opus-4-7", 1, 595, 0, 50_000);
        let alias = cache_aware_cost_cents("anthropic", "aura-claude-opus-4-7", 1, 595, 0, 50_000);
        assert_eq!(canonical, alias);
        assert!(canonical.is_some());
    }

    #[test]
    fn fireworks_kimi_discounts_cache_reads() {
        // new: (100_000 − 80_000) × 60 = 1_200_000
        // read: 80_000 × 30 = 2_400_000
        // output: 500 × 300 = 150_000
        // total: 3_750_000 / 1M = 3.75 × 1.20 = 4.5 → 5
        assert_eq!(
            cache_aware_cost_cents("fireworks", "aura-kimi-k2-5", 100_000, 500, 0, 80_000),
            Some(5)
        );
    }

    #[test]
    fn anthropic_mixed_cache_creation_and_read() {
        // Realistic: an agentic turn that both writes new content to cache
        // and reads existing cached content.
        // new: 5 × 500 = 2_500
        // write: 20_000 × 625 = 12_500_000
        // read: 80_000 × 50 = 4_000_000
        // output: 400 × 2500 = 1_000_000
        // total: 17_502_500 / 1M = 17.5025 × 1.20 = 21.003 → 21
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-opus-4-7", 5, 400, 20_000, 80_000),
            Some(21)
        );
    }

    #[test]
    fn anthropic_unknown_model_returns_none() {
        // Models not in the rates table fall through so z-billing's
        // per-model pricing applies (or default fallback).
        assert_eq!(
            cache_aware_cost_cents("anthropic", "claude-3-5-sonnet", 100, 100, 0, 1_000),
            None
        );
    }

    #[test]
    fn fireworks_unknown_model_returns_none() {
        assert_eq!(
            cache_aware_cost_cents(
                "fireworks",
                "accounts/fireworks/models/qwen2p5-coder-7b",
                100,
                100,
                0,
                1_000,
            ),
            None
        );
    }

    #[test]
    fn openai_cache_creation_billed_at_base_input_rate() {
        // Locks in pre-existing behavior: for OpenAI, cache_creation
        // tokens are billed at the same rate as new input (no premium).
        // new: 0 (all input was creation)
        // write: 10_000 × 500 = 5_000_000
        // output: 500 × 3000 = 1_500_000
        // total: 6_500_000 / 1M = 6.5 × 1.20 = 7.8 → 8
        assert_eq!(
            cache_aware_cost_cents("openai", "aura-gpt-5-5", 10_000, 500, 10_000, 0),
            Some(8)
        );
    }

    #[test]
    fn deepseek_via_fireworks_matches_direct_deepseek_cost() {
        // DeepSeek now bills under the "fireworks" provider; the cost must be
        // identical to the prior "deepseek"-provider result so the routing
        // change does not alter what users pay.
        for model in ["aura-deepseek-v4-pro", "aura-deepseek-v4-flash"] {
            let via_fireworks =
                cache_aware_cost_cents("fireworks", model, 1_000_000, 500_000, 0, 1_000_000);
            let via_deepseek =
                cache_aware_cost_cents("deepseek", model, 1_000_000, 500_000, 0, 1_000_000);
            assert_eq!(via_fireworks, via_deepseek, "cost parity for {model}");
            assert!(via_fireworks.is_some(), "rate present for {model}");
        }

        // The Fireworks upstream id resolves to the same rate as the aura id.
        assert_eq!(
            cache_aware_cost_cents(
                "fireworks",
                "accounts/fireworks/models/deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            cache_aware_cost_cents(
                "fireworks",
                "aura-deepseek-v4-pro",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
        );
    }

    #[test]
    fn fireworks_aura_aliases_resolve_to_same_rates() {
        let canonical = cache_aware_cost_cents(
            "fireworks",
            "accounts/fireworks/models/kimi-k2p5",
            100_000,
            500,
            0,
            80_000,
        );
        let alias = cache_aware_cost_cents("fireworks", "aura-kimi-k2-5", 100_000, 500, 0, 80_000);
        assert_eq!(canonical, alias);
        assert!(canonical.is_some());
    }
}
