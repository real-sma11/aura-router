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
struct DeepSeekRates {
    cache_hit_input_cents_per_million: f64,
    cache_miss_input_cents_per_million: f64,
    output_cents_per_million: f64,
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
    if provider != "deepseek" {
        return None;
    }

    let rates = deepseek_rates(model)?;
    if cache_creation_input_tokens == 0 && cache_read_input_tokens == 0 {
        return None;
    }

    let categorized_input = cache_creation_input_tokens.saturating_add(cache_read_input_tokens);
    let uncategorized_input = input_tokens.saturating_sub(categorized_input);
    let cache_miss_input_tokens = cache_creation_input_tokens.saturating_add(uncategorized_input);

    let total_cents = ((cache_miss_input_tokens as f64 * rates.cache_miss_input_cents_per_million)
        + (cache_read_input_tokens as f64 * rates.cache_hit_input_cents_per_million)
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

fn deepseek_rates(model: &str) -> Option<DeepSeekRates> {
    let model = model.strip_prefix("deepseek/").unwrap_or(model);

    match model {
        "aura-deepseek-v4-pro" | "deepseek-v4-pro" => Some(DeepSeekRates {
            cache_hit_input_cents_per_million: 14.5,
            cache_miss_input_cents_per_million: 174.0,
            output_cents_per_million: 348.0,
        }),
        "aura-deepseek-v4-flash" | "deepseek-v4-flash" | "deepseek-chat" | "deepseek-reasoner" => {
            Some(DeepSeekRates {
                cache_hit_input_cents_per_million: 2.8,
                cache_miss_input_cents_per_million: 14.0,
                output_cents_per_million: 28.0,
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
        assert_eq!(
            cache_aware_cost_cents(
                "fireworks",
                "aura-deepseek-v3-2",
                1_000_000,
                500_000,
                0,
                1_000_000,
            ),
            None
        );
    }
}
