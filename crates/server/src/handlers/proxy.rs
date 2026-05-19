//! LLM proxy handler — receives requests, checks credits, forwards to provider.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{anthropic_compat, billing, providers, stats, storage, stream};

use crate::state::AppState;

/// POST /v1/messages — Anthropic-compatible proxy endpoint.
///
/// Flow:
/// 1. Auth (JWT)
/// 2. Extract model from request body
/// 3. Resolve provider
/// 4. Pre-check credits via z-billing
/// 5. [ENRICHMENT HOOK — future: RAG, memory, prompt modification]
/// 6. Forward to provider with platform API key
/// 7. Debit credits + record usage (fire-and-forget)
/// 8. Return response
pub async fn messages(
    auth: AuthUser,
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, AppError> {
    let request_start = std::time::Instant::now();

    // Rate limit check
    if let Err(retry_after) = state.rate_limiter.check(&auth.user_id) {
        tracing::warn!(user_id = %auth.user_id, retry_after, "Rate limited");
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::RETRY_AFTER, retry_after.to_string()),
                (header::CONTENT_TYPE, "application/json".to_string()),
            ],
            Body::from(
                serde_json::json!({
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": format!("Too many requests. Retry after {retry_after} seconds.")
                    }
                })
                .to_string(),
            ),
        )
            .into_response());
    }

    // Parse just the model and stream fields from the request body
    let request_value: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let requested_model = request_value
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::BadRequest("Missing 'model' field".into()))?;

    let is_streaming = request_value
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Resolve the Aura-facing model id into an upstream provider/model pair.
    let resolved_model = providers::resolve_model(requested_model)
        .ok_or_else(|| AppError::BadRequest(format!("Unsupported model: {requested_model}")))?;
    let provider = resolved_model.provider;
    anthropic_compat::validate_request(provider, &request_value).map_err(AppError::BadRequest)?;

    // Public-guest requests (from the logged-out aura.ai surface) skip
    // billing entirely — cost is capped by the upstream rate limiter in
    // aura-os-server (3 turns/guest, 30/IP/day, global daily ceiling).
    let is_public_guest = auth.user_id == "public-guest";

    if !is_public_guest {
        // Pre-check credits using the z-billing pricing table when possible.
        let balance = billing::check_credits(
            &state.http_client,
            &state.z_billing_url,
            &state.z_billing_api_key,
            &auth.user_id,
            0,
            Some(provider.name()),
            Some(requested_model),
        )
        .await?;

        if !balance.sufficient {
            return Err(AppError::InsufficientCredits {
                balance: balance.balance_cents,
                required: balance.required_cents,
            });
        }
    }

    // Extract session context from custom headers (optional, for storage recording)
    let session_ctx = storage::SessionContext::from_headers(&headers);

    // Extract user content from the request for storage (last user message)
    let user_content = anthropic_compat::extract_last_user_text(&request_value);

    // [ENRICHMENT HOOK — v1: pass-through, future: RAG/memory/prompt modification]

    // Resolve API key for the provider
    let api_key = match resolved_model.provider {
        providers::Provider::Anthropic => state.anthropic_api_key.clone(),
        providers::Provider::OpenAi => state
            .openai_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("OpenAI provider not configured".into()))?,
        providers::Provider::Fireworks => state
            .fireworks_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("Fireworks provider not configured".into()))?,
        providers::Provider::DeepSeek => state
            .deepseek_api_key
            .clone()
            .ok_or_else(|| AppError::BadRequest("DeepSeek provider not configured".into()))?,
    };

    // Forward to provider
    let upstream_url = providers::provider_url(&resolved_model.provider);
    let upstream_headers = providers::provider_headers(&provider, &api_key)
        .ok_or_else(|| AppError::Internal("Invalid API key format".into()))?;
    let mut upstream_request_value = anthropic_compat::request_to_upstream(
        provider,
        resolved_model.upstream_model,
        &request_value,
    )
    .map_err(AppError::BadRequest)?;
    apply_provider_request_controls(
        provider,
        &mut upstream_request_value,
        &auth.user_id,
        resolved_model.upstream_model,
        session_ctx.as_ref(),
    );
    let upstream_body = serde_json::to_vec(&upstream_request_value)
        .map_err(|e| AppError::Internal(format!("Failed to encode upstream body: {e}")))?;

    let upstream_resp = state
        .http_client
        .post(upstream_url)
        .headers(upstream_headers)
        .body(upstream_body)
        .send()
        .await
        .map_err(|e| AppError::ProviderError(format!("Provider unreachable: {e}")))?;

    let upstream_status = upstream_resp.status();
    let provider_name = provider.name();

    // Normalize upstream failures into Anthropic-compatible error envelopes.
    if !upstream_status.is_success() {
        let error_body = upstream_resp.bytes().await.unwrap_or_default();
        return Ok(normalize_upstream_error(upstream_status, &error_body));
    }

    if is_streaming {
        return handle_streaming(
            auth,
            state,
            requested_model,
            provider,
            provider_name,
            upstream_resp,
            session_ctx,
            user_content,
            request_start,
        )
        .await;
    }

    handle_non_streaming(
        auth,
        state,
        requested_model,
        provider_name,
        upstream_resp,
        session_ctx,
        user_content,
        request_start,
    )
    .await
}

/// Handle non-streaming response: read full body, extract usage, debit, return.
async fn handle_non_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    provider_name: &str,
    upstream_resp: reqwest::Response,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
    request_start: std::time::Instant,
) -> Result<Response, AppError> {
    let response_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to read provider response: {e}")))?;

    let upstream_value: serde_json::Value = serde_json::from_slice(&response_bytes)
        .map_err(|e| AppError::ProviderError(format!("Provider returned invalid JSON: {e}")))?;
    let response_value = anthropic_compat::response_from_upstream(
        provider_from_name(provider_name),
        model,
        &upstream_value,
    )
    .map_err(AppError::ProviderError)?;
    let normalized_response_bytes = serde_json::to_vec(&response_value)
        .map_err(|e| AppError::Internal(format!("Failed to encode normalized response: {e}")))?;

    let input_tokens = response_value
        .pointer("/usage/input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let output_tokens = response_value
        .pointer("/usage/output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_input_tokens = response_value
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read_input_tokens = response_value
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let org_id_ref = session_ctx.as_ref().and_then(|c| c.org_id.as_deref());
    let project_id_ref = session_ctx.as_ref().and_then(|c| c.project_id.as_deref());
    let agent_id_ref = session_ctx.as_ref().and_then(|c| c.project_agent_id.as_deref());

    let duration_ms = request_start.elapsed().as_millis() as u64;

    spawn_post_request_tasks(
        &state,
        &auth.user_id,
        org_id_ref,
        project_id_ref,
        agent_id_ref,
        provider_name,
        model,
        input_tokens,
        output_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        duration_ms,
    );

    // Store messages to aura-storage if session context is present
    if let Some(ctx) = session_ctx {
        if let (Some(ref storage_url), Some(ref storage_token)) =
            (&state.aura_storage_url, &state.aura_storage_token)
        {
            let assistant_content = anthropic_compat::extract_response_text(&response_value);

            let client = state.http_client.clone();
            let url = storage_url.clone();
            let token = storage_token.clone();
            let user_id = auth.user_id.clone();
            tokio::spawn(async move {
                storage::store_events(
                    &client,
                    &url,
                    &token,
                    &ctx,
                    &user_id,
                    &user_content,
                    &assistant_content,
                    None,
                    input_tokens,
                    output_tokens,
                )
                .await;
            });
        }
    }

    let max_tokens = providers::max_context_tokens(model);
    let context_usage = if max_tokens > 0 {
        input_tokens as f64 / max_tokens as f64
    } else {
        0.0
    };

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (
                header::HeaderName::from_static("x-context-usage"),
                format!("{context_usage:.4}"),
            ),
            (
                header::HeaderName::from_static("x-model-max-tokens"),
                max_tokens.to_string(),
            ),
        ],
        Body::from(normalized_response_bytes),
    )
        .into_response())
}

/// Handle streaming response: tee SSE stream to client while capturing billing data.
async fn handle_streaming(
    auth: AuthUser,
    state: AppState,
    model: &str,
    provider: providers::Provider,
    provider_name: &str,
    upstream_resp: reqwest::Response,
    session_ctx: Option<storage::SessionContext>,
    user_content: String,
    request_start: std::time::Instant,
) -> Result<Response, AppError> {
    let model_owned = model.to_string();
    let provider_owned = provider_name.to_string();
    let (tee_stream, usage_rx) = stream::proxy_stream(provider, model, upstream_resp);

    // Spawn task to handle billing + storage after stream completes
    let billing_state = state.clone();
    let user_id = auth.user_id.clone();
    let stream_org_id = session_ctx.as_ref().and_then(|c| c.org_id.clone());
    let stream_project_id = session_ctx.as_ref().and_then(|c| c.project_id.clone());
    let stream_agent_id = session_ctx.as_ref().and_then(|c| c.project_agent_id.clone());
    tokio::spawn(async move {
        if let Ok(usage) = usage_rx.await {
            let duration_ms = request_start.elapsed().as_millis() as u64;
            let model = usage.model.as_deref().unwrap_or(&model_owned);
            spawn_post_request_tasks(
                &billing_state,
                &user_id,
                stream_org_id.as_deref(),
                stream_project_id.as_deref(),
                stream_agent_id.as_deref(),
                &provider_owned,
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
                duration_ms,
            );

            // Store messages to aura-storage if session context is present
            if let Some(ctx) = session_ctx {
                if let (Some(ref storage_url), Some(ref storage_token)) = (
                    &billing_state.aura_storage_url,
                    &billing_state.aura_storage_token,
                ) {
                    storage::store_events(
                        &billing_state.http_client,
                        storage_url,
                        storage_token,
                        &ctx,
                        &user_id,
                        &user_content,
                        "[streamed response]",
                        None,
                        usage.input_tokens,
                        usage.output_tokens,
                    )
                    .await;
                }
            }
        }
    });

    let max_tokens = providers::max_context_tokens(model);
    let body = Body::from_stream(tee_stream);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream".to_string()),
            (header::CACHE_CONTROL, "no-cache".to_string()),
            (
                header::HeaderName::from_static("x-accel-buffering"),
                "no".to_string(),
            ),
            (
                header::HeaderName::from_static("x-model-max-tokens"),
                max_tokens.to_string(),
            ),
        ],
        body,
    )
        .into_response())
}

fn provider_from_name(provider_name: &str) -> providers::Provider {
    match provider_name {
        "openai" => providers::Provider::OpenAi,
        "fireworks" => providers::Provider::Fireworks,
        "deepseek" => providers::Provider::DeepSeek,
        _ => providers::Provider::Anthropic,
    }
}

fn apply_provider_request_controls(
    provider: providers::Provider,
    upstream_body: &mut serde_json::Value,
    user_id: &str,
    upstream_model: &str,
    session_ctx: Option<&storage::SessionContext>,
) {
    if provider != providers::Provider::OpenAi {
        return;
    }

    let Some(body) = upstream_body.as_object_mut() else {
        return;
    };
    body.entry("prompt_cache_key").or_insert_with(|| {
        serde_json::Value::String(openai_prompt_cache_key(
            user_id,
            upstream_model,
            session_ctx,
        ))
    });
}

fn openai_prompt_cache_key(
    user_id: &str,
    upstream_model: &str,
    session_ctx: Option<&storage::SessionContext>,
) -> String {
    let model = sanitize_cache_key_component(upstream_model);
    match session_ctx {
        Some(ctx) => {
            let principal = ctx
                .org_id
                .as_deref()
                .map(|org_id| format!("org:{}", sanitize_cache_key_component(org_id)))
                .unwrap_or_else(|| format!("user:{}", sanitize_cache_key_component(user_id)));
            // Cache key components fall back to "none" placeholders when
            // headers are absent. The user_id / org_id principal still
            // segregates per-tenant, so missing project/agent ids share
            // a bucket within a single principal — acceptable.
            format!(
                "aura:v1:{principal}:project:{}:agent:{}:model:{model}",
                sanitize_cache_key_component(ctx.project_id.as_deref().unwrap_or("none")),
                sanitize_cache_key_component(
                    ctx.project_agent_id.as_deref().unwrap_or("none"),
                ),
            )
        }
        None => format!(
            "aura:v1:user:{}:model:{model}",
            sanitize_cache_key_component(user_id)
        ),
    }
}

fn sanitize_cache_key_component(value: &str) -> String {
    let sanitized: String = value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '@') {
                ch
            } else {
                '_'
            }
        })
        .take(96)
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn normalize_upstream_error(status: StatusCode, error_body: &[u8]) -> Response {
    let message = extract_upstream_error_message(error_body)
        .unwrap_or_else(|| format!("Upstream provider returned HTTP {}", status.as_u16()));

    (
        status,
        [(header::CONTENT_TYPE, "application/json".to_string())],
        Body::from(
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": anthropic_error_type(status),
                    "message": message,
                }
            })
            .to_string(),
        ),
    )
        .into_response()
}

fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.as_u16() == 529 => "overloaded_error",
        _ => "api_error",
    }
}

fn extract_upstream_error_message(error_body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(error_body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    value
                        .pointer("/message")
                        .and_then(serde_json::Value::as_str)
                })
                .or_else(|| value.pointer("/error").and_then(serde_json::Value::as_str))
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            std::str::from_utf8(error_body)
                .ok()
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .map(str::to_string)
        })
}

/// Fire-and-forget tasks: debit z-billing + record to aura-network.
#[allow(clippy::too_many_arguments)]
fn spawn_post_request_tasks(
    state: &AppState,
    user_id: &str,
    org_id: Option<&str>,
    project_id: Option<&str>,
    agent_id: Option<&str>,
    provider_name: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
    duration_ms: u64,
) {
    let event_id = uuid::Uuid::new_v4().to_string();
    let model_owned = model.to_string();
    let user_id_owned = user_id.to_string();
    let provider_owned = provider_name.to_string();
    let org_id_owned = org_id.map(String::from);
    let project_id_owned = project_id.map(String::from);
    let agent_id_owned = agent_id.map(String::from);
    let cost_cents = billing::cache_aware_cost_cents(
        provider_name,
        model,
        input_tokens,
        output_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
    );

    // Debit z-billing (skip for public-guest — no billing account)
    if user_id != "public-guest" {
        let client = state.http_client.clone();
        let billing_url = state.z_billing_url.clone();
        let billing_key = state.z_billing_api_key.clone();
        let user_id = user_id_owned.clone();
        let model = model_owned.clone();
        let provider = provider_owned.clone();
        tokio::spawn(async move {
            if let Err(e) = billing::report_usage(
                &client,
                &billing_url,
                &billing_key,
                &event_id,
                &user_id,
                &provider,
                &model,
                input_tokens,
                output_tokens,
                cost_cents,
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to debit credits via z-billing");
            }
        });
    }

    // Record to aura-network
    if let (Some(ref network_url), Some(ref network_token)) =
        (&state.aura_network_url, &state.aura_network_token)
    {
        let client = state.http_client.clone();
        let url = network_url.clone();
        let token = network_token.clone();
        let user_id = user_id_owned;
        let model = model_owned;
        tokio::spawn(async move {
            stats::record_usage(
                &client,
                &url,
                &token,
                &user_id,
                org_id_owned.as_deref(),
                project_id_owned.as_deref(),
                agent_id_owned.as_deref(),
                &model,
                input_tokens,
                output_tokens,
                // Prefer the cache-aware cost (model-specific rate
                // accounting for cache create/read tokens) computed
                // above for z-billing. Fall back to the legacy flat
                // estimate so the cost field stays populated for
                // providers/models that lack a price entry.
                cost_cents
                    .map(|c| c as f64 / 100.0)
                    .unwrap_or((input_tokens + output_tokens) as f64 * 0.00001),
                duration_ms,
            )
            .await;
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{router, state::AppState};
    use aura_router_auth::{InternalToken, TokenValidator};
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use axum::routing::post;
    use axum::{Json, Router};
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use serde::Serialize;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    const SELF_SIGNED_KID: &str = "jFNXMnFjGrSoDafnLQBohoCNalWcFcTjnKEbkRzWFBHyYJFikdLMHP";

    #[derive(Debug, Serialize)]
    struct TestClaims {
        id: String,
    }

    async fn start_mock_billing() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route(
                "/v1/usage/check",
                post(|| async {
                    Json(json!({
                        "sufficient": true,
                        "balance_cents": 1_000_000,
                        "required_cents": 1,
                    }))
                }),
            )
            .route(
                "/v1/usage",
                post(|| async {
                    Json(json!({
                        "success": true,
                        "balance_cents": 999_999,
                        "cost_cents": 1,
                        "transaction_id": "txn_test",
                    }))
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (url, handle)
    }

    async fn start_recording_billing(
        check_response: serde_json::Value,
    ) -> (
        String,
        Arc<Mutex<Vec<serde_json::Value>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let recorded_requests = Arc::new(Mutex::new(Vec::new()));
        let recorded_for_check = Arc::clone(&recorded_requests);
        let recorded_for_usage = Arc::clone(&recorded_requests);
        let check_response_for_route = check_response.clone();

        let app = Router::new()
            .route(
                "/v1/usage/check",
                post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                    let recorded_for_check = Arc::clone(&recorded_for_check);
                    let check_response = check_response_for_route.clone();
                    async move {
                        recorded_for_check.lock().unwrap().push(body);
                        Json(check_response)
                    }
                }),
            )
            .route(
                "/v1/usage",
                post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                    let recorded_for_usage = Arc::clone(&recorded_for_usage);
                    async move {
                        let cost_cents = body
                            .get("cost_cents")
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(1);
                        recorded_for_usage.lock().unwrap().push(body);
                        Json(json!({
                            "success": true,
                            "balance_cents": 999_999,
                            "cost_cents": cost_cents,
                            "transaction_id": "txn_test",
                        }))
                    }
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (url, recorded_requests, handle)
    }

    fn test_jwt(secret: &str, user_id: &str) -> String {
        let mut header = Header::new(Algorithm::HS256);
        header.kid = Some(SELF_SIGNED_KID.to_string());
        encode(
            &header,
            &TestClaims {
                id: user_id.to_string(),
            },
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("test jwt")
    }

    fn test_state(
        cookie_secret: &str,
        billing_url: String,
        anthropic_api_key: String,
        openai_api_key: Option<String>,
        fireworks_api_key: Option<String>,
    ) -> AppState {
        AppState {
            validator: TokenValidator::new(
                "example.auth0.test".to_string(),
                "aura-router-tests".to_string(),
                cookie_secret.to_string(),
            ),
            internal_token: InternalToken("internal-test-token".to_string()),
            public_guest_token: None,
            http_client: reqwest::Client::new(),
            rate_limiter: std::sync::Arc::new(aura_router_proxy::rate_limit::RateLimiter::new(
                120, 60,
            )),
            anthropic_api_key,
            openai_api_key,
            fireworks_api_key,
            deepseek_api_key: None,
            google_api_key: None,
            tripo_api_key: None,
            ark_api_key: None,
            z_billing_url: billing_url,
            z_billing_api_key: "billing-test-key".to_string(),
            aura_network_url: None,
            aura_network_token: None,
            aura_storage_url: None,
            aura_storage_token: None,
            s3_config: None,
            watermark_bytes: None,
        }
    }

    #[test]
    fn openai_prompt_cache_key_uses_stable_project_agent_scope() {
        let session = aura_router_proxy::storage::SessionContext {
            session_id: Some("session-ignored".to_string()),
            project_agent_id: Some("agent:123".to_string()),
            project_id: Some("project 456".to_string()),
            org_id: Some("org-789".to_string()),
        };

        let key = super::openai_prompt_cache_key("user-1", "gpt-5.5", Some(&session));

        assert_eq!(
            key,
            "aura:v1:org:org-789:project:project_456:agent:agent_123:model:gpt-5.5"
        );
        assert!(!key.contains("session-ignored"));
    }

    #[test]
    fn openai_prompt_cache_key_falls_back_to_user_scope() {
        let key = super::openai_prompt_cache_key("user@example.com", "gpt-5.5", None);

        assert_eq!(key, "aura:v1:user:user@example.com:model:gpt-5.5");
    }

    #[test]
    fn openai_provider_request_controls_add_prompt_cache_key() {
        let session = aura_router_proxy::storage::SessionContext {
            session_id: Some("session-ignored".to_string()),
            project_agent_id: Some("agent-a".to_string()),
            project_id: Some("project-a".to_string()),
            org_id: Some("org-a".to_string()),
        };
        let mut upstream = json!({
            "model": "gpt-5.5",
            "messages": []
        });

        super::apply_provider_request_controls(
            aura_router_proxy::providers::Provider::OpenAi,
            &mut upstream,
            "user-a",
            "gpt-5.5",
            Some(&session),
        );

        assert_eq!(
            upstream["prompt_cache_key"],
            "aura:v1:org:org-a:project:project-a:agent:agent-a:model:gpt-5.5"
        );
        assert!(upstream.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn normalizes_upstream_json_errors_to_anthropic_shape() {
        let response = super::normalize_upstream_error(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"tools[0] is invalid"}}"#,
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "tools[0] is invalid");
    }

    #[tokio::test]
    async fn uses_requested_aura_model_for_model_aware_credit_check() {
        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-credit-check");
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": false,
            "balance_cents": 1,
            "required_cents": 3,
        }))
        .await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            Some("unused".to_string()),
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-gpt-5-5",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["required_cents"], 0);
        assert_eq!(requests[0]["provider"], "openai");
        assert_eq!(requests[0]["model"], "aura-gpt-5-5");

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value = serde_json::from_slice(&body).expect("json response");
        assert_eq!(response["error"]["code"], "INSUFFICIENT_CREDITS");
        assert!(response["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("required=3"));
    }

    #[tokio::test]
    async fn uses_deepseek_provider_for_model_aware_credit_check() {
        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-deepseek-credit-check");
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": false,
            "balance_cents": 1,
            "required_cents": 3,
        }))
        .await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            None,
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-deepseek-v4-flash",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["provider"], "deepseek");
        assert_eq!(requests[0]["model"], "aura-deepseek-v4-flash");
    }

    #[tokio::test]
    async fn reports_deepseek_cache_aware_usage_cost_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-deepseek-billing",
            None,
            None,
            None,
            "deepseek",
            "aura-deepseek-v4-flash",
            1_000_000,
            500_000,
            0,
            1_000_000,
            123,
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["user_id"], "user-deepseek-billing");
        assert_eq!(requests[0]["cost_cents"], 20);
        assert_eq!(requests[0]["metric"]["provider"], "deepseek");
        assert_eq!(requests[0]["metric"]["model"], "aura-deepseek-v4-flash");
        assert_eq!(requests[0]["metric"]["input_tokens"], 1_000_000);
        assert_eq!(requests[0]["metric"]["output_tokens"], 500_000);
    }

    #[tokio::test]
    async fn reports_openai_cache_aware_usage_cost_to_billing() {
        let cookie_secret = "test-cookie-secret";
        let (billing_url, recorded_requests, _billing_handle) = start_recording_billing(json!({
            "sufficient": true,
            "balance_cents": 1_000_000,
            "required_cents": 1,
        }))
        .await;
        let state = test_state(cookie_secret, billing_url, "unused".to_string(), None, None);

        super::spawn_post_request_tasks(
            &state,
            "user-openai-billing",
            Some("org-openai"),
            Some("project-openai"),
            None,
            "openai",
            "aura-gpt-5-5",
            1_000_000,
            500_000,
            0,
            1_000_000,
            123,
        );

        for _ in 0..20 {
            if !recorded_requests.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        let requests = recorded_requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["user_id"], "user-openai-billing");
        assert_eq!(requests[0]["cost_cents"], 1860);
        assert_eq!(requests[0]["metric"]["provider"], "openai");
        assert_eq!(requests[0]["metric"]["model"], "aura-gpt-5-5");
        assert_eq!(requests[0]["metric"]["input_tokens"], 1_000_000);
        assert_eq!(requests[0]["metric"]["output_tokens"], 500_000);
    }

    #[tokio::test]
    async fn normalizes_unparseable_upstream_errors_without_passthrough() {
        let response = super::normalize_upstream_error(StatusCode::BAD_GATEWAY, b"");

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");

        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "api_error");
        assert_eq!(
            body["error"]["message"],
            "Upstream provider returned HTTP 502"
        );
    }

    #[tokio::test]
    async fn anthropic_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(anthropic_api_key) = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping Anthropic live smoke test because ANTHROPIC_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-anthropic-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            anthropic_api_key,
            None,
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-claude-sonnet-4-6",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly ANTHROPIC_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-claude-sonnet-4-6");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("ANTHROPIC_OK"),
            "expected live Anthropic response to contain ANTHROPIC_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Anthropic input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Anthropic output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn openai_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(openai_api_key) = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping OpenAI live smoke test because OPENAI_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-openai-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            Some(openai_api_key),
            None,
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-gpt-5-4-mini",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly OPENAI_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-gpt-5-4-mini");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("OPENAI_OK"),
            "expected live OpenAI response to contain OPENAI_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected OpenAI input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected OpenAI output token count to be populated: {response}"
        );
    }

    #[tokio::test]
    async fn fireworks_live_smoke_for_aura_managed_model() {
        dotenvy::dotenv().ok();
        let Some(fireworks_api_key) = std::env::var("FIREWORKS_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping Fireworks live smoke test because FIREWORKS_API_KEY is missing");
            return;
        };

        let cookie_secret = "test-cookie-secret";
        let jwt = test_jwt(cookie_secret, "user-fireworks-smoke");
        let (billing_url, _billing_handle) = start_mock_billing().await;

        let app = router::create_router().with_state(test_state(
            cookie_secret,
            billing_url,
            "unused".to_string(),
            None,
            Some(fireworks_api_key),
        ));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(header::AUTHORIZATION, format!("Bearer {jwt}"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "aura-kimi-k2-5",
                    "max_tokens": 32,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Reply with exactly FIREWORKS_OK and nothing else."
                                }
                            ]
                        }
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("response bytes");
        let response: serde_json::Value =
            serde_json::from_slice(&bytes).expect("normalized anthropic response");
        assert_eq!(response["model"], "aura-kimi-k2-5");
        let text = response["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    (block.get("type").and_then(|v| v.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|v| v.as_str()))
                        .flatten()
                })
            })
            .unwrap_or_default();
        assert!(
            text.contains("FIREWORKS_OK"),
            "expected live Fireworks response to contain FIREWORKS_OK, got: {text}"
        );
        assert!(
            response["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Fireworks input token count to be populated: {response}"
        );
        assert!(
            response["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "expected Fireworks output token count to be populated: {response}"
        );
    }
}
