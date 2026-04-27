//! aura-network usage recording client.

/// Record token usage to aura-network (fire-and-forget).
///
/// Calls POST /internal/usage with X-Internal-Token. Any of `org_id`,
/// `project_id`, `agent_id` may be `None`; the receiver stores `null`
/// and aggregations scoped by those columns simply exclude the row.
///
/// IMPORTANT: `agent_id` is currently swallowed and NOT sent to
/// aura-network — passing it would trigger
/// `token_usage_daily_agent_id_fkey` FK violations. The header
/// `x-aura-agent-id` carries aura-code's `project_agents.id`, but
/// aura-network's FK references its own `agents` table — different
/// tables in different services. Until proper id translation lands,
/// we keep the legacy behaviour of sending `agentId: null` so the
/// row inserts cleanly. Per-agent attribution is a follow-up.
/// Errors are logged but do not block the response.
#[allow(clippy::too_many_arguments)]
pub async fn record_usage(
    client: &reqwest::Client,
    network_url: &str,
    token: &str,
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    _agent_id: Option<&str>,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    duration_ms: u64,
) {
    let url = format!("{network_url}/internal/usage");

    let result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "orgId": org_id,
            "userId": user_id,
            "zeroUserId": user_id,
            "agentId": serde_json::Value::Null,
            "projectId": project_id,
            "model": model,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "estimatedCostUsd": cost_usd,
            "durationMs": duration_ms
        }))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(user_id = %user_id, model = %model, "Usage recorded to aura-network");
        }
        Ok(resp) => {
            tracing::warn!(
                status = %resp.status(),
                "Failed to record usage to aura-network"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-network for usage recording");
        }
    }
}
