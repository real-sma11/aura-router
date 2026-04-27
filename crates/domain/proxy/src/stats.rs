//! aura-network usage recording client.

/// Record token usage to aura-network (fire-and-forget).
///
/// Calls POST /internal/usage with X-Internal-Token. Any of `org_id`,
/// `project_id`, `agent_id`, `task_id` may be `None`; the receiver stores
/// `null` and aggregations scoped by those columns simply exclude the row.
/// Errors are logged but do not block the response.
#[allow(clippy::too_many_arguments)]
pub async fn record_usage(
    client: &reqwest::Client,
    network_url: &str,
    token: &str,
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    agent_id: Option<&str>,
    task_id: Option<&str>,
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
            "agentId": agent_id,
            "projectId": project_id,
            "taskId": task_id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    async fn spawn_recorder() -> (String, Arc<Mutex<Option<serde_json::Value>>>) {
        let recorded: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let recorded_clone = Arc::clone(&recorded);

        let app = Router::new().route(
            "/internal/usage",
            post(move |Json(body): Json<serde_json::Value>| {
                let recorded = Arc::clone(&recorded_clone);
                async move {
                    *recorded.lock().unwrap() = Some(body);
                    axum::http::StatusCode::OK
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (url, recorded)
    }

    #[tokio::test]
    async fn record_usage_payload_carries_all_attribution_ids() {
        let (url, recorded) = spawn_recorder().await;
        let client = reqwest::Client::new();

        record_usage(
            &client,
            &url,
            "test-token",
            "user-1",
            Some("org-1"),
            Some("proj-1"),
            Some("agent-1"),
            Some("task-1"),
            "claude-test",
            100,
            50,
            0.0015,
            1234,
        )
        .await;

        let payload = recorded
            .lock()
            .unwrap()
            .clone()
            .expect("recorder did not capture payload");
        assert_eq!(payload["userId"], "user-1");
        assert_eq!(payload["orgId"], "org-1");
        assert_eq!(payload["projectId"], "proj-1");
        assert_eq!(payload["agentId"], "agent-1");
        assert_eq!(payload["taskId"], "task-1");
        assert_eq!(payload["model"], "claude-test");
        assert_eq!(payload["inputTokens"], 100);
        assert_eq!(payload["outputTokens"], 50);
        assert_eq!(payload["estimatedCostUsd"], 0.0015);
        assert_eq!(payload["durationMs"], 1234);
    }

    #[tokio::test]
    async fn record_usage_sends_null_for_absent_ids() {
        let (url, recorded) = spawn_recorder().await;
        let client = reqwest::Client::new();

        record_usage(
            &client,
            &url,
            "test-token",
            "user-1",
            None,
            None,
            None,
            None,
            "model-x",
            10,
            5,
            0.0,
            7,
        )
        .await;

        let payload = recorded.lock().unwrap().clone().expect("payload");
        assert!(payload["orgId"].is_null());
        assert!(payload["projectId"].is_null());
        assert!(payload["agentId"].is_null());
        assert!(payload["taskId"].is_null());
        assert_eq!(payload["durationMs"], 7);
    }
}
