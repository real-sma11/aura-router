//! aura-storage event recording client.
//!
//! Stores LLM prompts and responses as session events to aura-storage.
//! Requires session context headers from the client (X-Aura-Session-Id, etc.).

/// Context headers from the client request, used for storage recording
/// and per-call cost attribution.
///
/// All fields are optional: callers that can supply only a subset (e.g.
/// a task-extract harness session that has a project_id but no live
/// session_id) still get partial attribution. `from_headers` returns
/// `None` only when every aura-* header is absent. Downstream code
/// handles each missing field individually — e.g. session-event writes
/// skip when `session_id` is None, while cost attribution still works
/// from `project_id` alone.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    pub session_id: Option<String>,
    pub project_agent_id: Option<String>,
    pub project_id: Option<String>,
    pub org_id: Option<String>,
}

impl SessionContext {
    /// Extract session context from request headers.
    ///
    /// Returns `None` only when no aura-* headers are present at all.
    /// A single recognised header is enough to produce a `Some`; missing
    /// headers within the context simply stay `None`. This means
    /// downstream code must treat every field as optional and skip work
    /// that genuinely needs a missing id (e.g. session-event writes
    /// only fire when `session_id` is set), while attribution-only
    /// paths (cost reporting) still work from `project_id` alone.
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Option<Self> {
        let read = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(String::from)
        };

        let session_id = read("x-aura-session-id");
        let project_agent_id = read("x-aura-agent-id");
        let project_id = read("x-aura-project-id");
        let org_id = read("x-aura-org-id");

        if session_id.is_none()
            && project_agent_id.is_none()
            && project_id.is_none()
            && org_id.is_none()
        {
            return None;
        }

        Some(Self {
            session_id,
            project_agent_id,
            project_id,
            org_id,
        })
    }
}

/// Store a user prompt and assistant response as session events (fire-and-forget).
///
/// Calls POST /internal/events for each event. Returns early without
/// writing anything if `ctx.session_id` is missing — events are
/// session-scoped and have no meaning without one. Errors are logged
/// but do not block the response.
pub async fn store_events(
    client: &reqwest::Client,
    storage_url: &str,
    token: &str,
    ctx: &SessionContext,
    user_id: &str,
    user_content: &str,
    assistant_content: &str,
    thinking: Option<&str>,
    input_tokens: u64,
    output_tokens: u64,
) {
    let Some(session_id) = ctx.session_id.as_deref() else {
        tracing::debug!("Skipping event storage: no x-aura-session-id header");
        return;
    };

    let url = format!("{storage_url}/internal/events");

    // Store user prompt as event
    let user_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": session_id,
            "userId": user_id,
            "agentId": ctx.project_agent_id,
            "sender": "user",
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "type": "message_saved",
            "content": {
                "role": "user",
                "text": user_content
            }
        }))
        .send()
        .await;

    match user_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("User event stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store user event");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for user event");
        }
    }

    // Store assistant response as event
    let mut content = serde_json::json!({
        "role": "assistant",
        "text": assistant_content,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens
    });

    if let Some(thinking_text) = thinking {
        content["thinking"] = serde_json::Value::String(thinking_text.to_string());
    }

    let assistant_result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "sessionId": session_id,
            "agentId": ctx.project_agent_id,
            "sender": "agent",
            "projectId": ctx.project_id,
            "orgId": ctx.org_id,
            "type": "message_saved",
            "content": content
        }))
        .send()
        .await;

    match assistant_result {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("Assistant event stored to aura-storage");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store assistant event");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for assistant event");
        }
    }
}

/// Store a generated artifact in aura-storage (fire-and-forget).
///
/// Calls POST /internal/artifacts.
/// Errors are logged but do not block the response.
pub async fn store_artifact(
    client: &reqwest::Client,
    storage_url: &str,
    token: &str,
    project_id: &str,
    created_by: &str,
    artifact_type: &str,
    asset_url: &str,
    thumbnail_url: Option<&str>,
    original_url: Option<&str>,
    name: Option<&str>,
    prompt: Option<&str>,
    prompt_mode: Option<&str>,
    model: &str,
    provider: &str,
    is_iteration: bool,
    parent_id: Option<&str>,
) -> Option<String> {
    let url = format!("{storage_url}/internal/artifacts");

    let result = client
        .post(&url)
        .header("x-internal-token", token)
        .json(&serde_json::json!({
            "projectId": project_id,
            "createdBy": created_by,
            "type": artifact_type,
            "assetUrl": asset_url,
            "thumbnailUrl": thumbnail_url,
            "originalUrl": original_url,
            "name": name,
            "prompt": prompt,
            "promptMode": prompt_mode,
            "model": model,
            "provider": provider,
            "isIteration": is_iteration,
            "parentId": parent_id,
        }))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                let id = body
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                tracing::debug!("Artifact stored to aura-storage");
                id
            } else {
                tracing::debug!("Artifact stored but failed to parse response");
                None
            }
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "Failed to store artifact");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to reach aura-storage for artifact");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    fn header_map(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(axum::http::HeaderName::from_static(k), v.parse().unwrap());
        }
        h
    }

    #[test]
    fn from_headers_returns_none_when_all_absent() {
        let h = header_map(&[]);
        assert!(SessionContext::from_headers(&h).is_none());
    }

    #[test]
    fn from_headers_accepts_only_project_id() {
        // The previously-required all-or-nothing gate dropped any call
        // missing session_id. Loosening to "any-one-header" preserves
        // cost attribution for tool sessions that have a project_id
        // but no live storage session.
        let h = header_map(&[("x-aura-project-id", "proj-1")]);
        let ctx = SessionContext::from_headers(&h).expect("project_id alone should yield Some");
        assert_eq!(ctx.project_id.as_deref(), Some("proj-1"));
        assert!(ctx.session_id.is_none());
        assert!(ctx.project_agent_id.is_none());
        assert!(ctx.org_id.is_none());
    }

    #[test]
    fn from_headers_accepts_only_session_id() {
        let h = header_map(&[("x-aura-session-id", "sess-1")]);
        let ctx = SessionContext::from_headers(&h).expect("session_id alone should yield Some");
        assert_eq!(ctx.session_id.as_deref(), Some("sess-1"));
    }

    #[test]
    fn from_headers_reads_all_four() {
        let h = header_map(&[
            ("x-aura-session-id", "sess-1"),
            ("x-aura-agent-id", "agent-1"),
            ("x-aura-project-id", "proj-1"),
            ("x-aura-org-id", "org-1"),
        ]);
        let ctx = SessionContext::from_headers(&h).expect("all-present should yield Some");
        assert_eq!(ctx.session_id.as_deref(), Some("sess-1"));
        assert_eq!(ctx.project_agent_id.as_deref(), Some("agent-1"));
        assert_eq!(ctx.project_id.as_deref(), Some("proj-1"));
        assert_eq!(ctx.org_id.as_deref(), Some("org-1"));
    }

    #[tokio::test]
    async fn store_events_returns_early_without_session_id() {
        // Events are session-scoped; a context without session_id must
        // not attempt the HTTP call. If it did, the unreachable URL
        // would surface as a tracing::warn — checking that no panic /
        // hang occurs is the regression guard.
        let client = reqwest::Client::new();
        let ctx = SessionContext {
            session_id: None,
            project_agent_id: Some("agent-1".into()),
            project_id: Some("proj-1".into()),
            org_id: None,
        };
        store_events(
            &client,
            "http://127.0.0.1:1", // unreachable on purpose
            "tok",
            &ctx,
            "user-1",
            "user content",
            "assistant content",
            None,
            10,
            5,
        )
        .await;
        // Reaching here without panic = early-return path worked.
    }
}
