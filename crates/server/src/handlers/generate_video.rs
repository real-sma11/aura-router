//! Video generation handler — generates videos via Google Veo or BytePlus
//! Seedance, uploads to S3.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use aura_router_auth::AuthUser;
use aura_router_core::AppError;
use aura_router_proxy::{billing, seedance, storage, veo};
use aura_router_proxy::video_types::{VideoGenConfig, VideoMeta, VideoStreamEvent};

use crate::state::AppState;

/// POST /v1/generate-video/stream — Stream video generation with SSE.
pub async fn generate_video_stream(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<veo::GenerateVideoRequest>,
) -> Result<Response, AppError> {
    if input.prompt.trim().is_empty() {
        return Err(AppError::BadRequest("Prompt must not be empty".into()));
    }

    let s3_config = state
        .s3_config
        .clone()
        .ok_or_else(|| AppError::Internal("Video generation not configured (S3)".into()))?;

    if let Err(retry_after) = state.rate_limiter.check(&auth.user_id) {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, retry_after.to_string())],
            axum::body::Body::from(
                serde_json::json!({
                    "error": { "code": "RATE_LIMITED", "message": format!("Retry after {retry_after} seconds.") }
                })
                .to_string(),
            ),
        )
            .into_response());
    }

    let is_seedance = seedance::is_seedance_model(&input.model);

    // Resolve provider-specific API key
    let api_key = if is_seedance {
        state
            .ark_api_key
            .as_ref()
            .ok_or_else(|| AppError::Internal("Video generation not configured (ARK API key)".into()))?
            .clone()
    } else {
        state
            .google_api_key
            .as_ref()
            .ok_or_else(|| AppError::Internal("Video generation not configured (Google API key)".into()))?
            .clone()
    };

    // Calculate cost (provider-specific pricing)
    let cost_cents = if is_seedance {
        seedance::cost_cents(&input.model, &input.resolution, &input.aspect_ratio, input.duration_seconds)
    } else {
        veo::cost_cents(&input.model, &input.resolution, input.duration_seconds)
    };

    let balance = billing::check_credits(
        &state.http_client,
        &state.z_billing_url,
        &state.z_billing_api_key,
        &auth.user_id,
        cost_cents,
        None,
        None,
    )
    .await?;

    if !balance.sufficient {
        return Err(AppError::InsufficientCredits {
            balance: balance.balance_cents,
            required: cost_cents,
        });
    }

    let model = input.model.clone();
    let prompt = input.prompt.clone();
    let aspect_ratio = input.aspect_ratio.clone();
    let duration_seconds = input.duration_seconds;
    let resolution = input.resolution.clone();
    let generate_audio = input.generate_audio;

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<VideoStreamEvent>(32);

    let gen_state = state.clone();
    let gen_user_id = auth.user_id.clone();
    let gen_project_id = input.project_id.clone();
    let gen_name = input.name.clone();

    if is_seedance {
        tokio::spawn(async move {
            let provider = "byteplus";

            let _ = event_tx
                .send(VideoStreamEvent::Start {
                    ts: chrono::Utc::now().to_rfc3339(),
                })
                .await;

            // Submit task to Seedance
            let task_id = match seedance::create_task(
                &gen_state.http_client,
                &api_key,
                &prompt,
                &model,
                &aspect_ratio,
                duration_seconds,
                &resolution,
                generate_audio,
            )
            .await
            {
                Ok(id) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Progress {
                            percent: 5,
                            message: "Video generation started...".to_string(),
                        })
                        .await;
                    id
                }
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "SUBMIT_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            // Poll for completion
            let video_url_temp = match seedance::poll_task(
                &gen_state.http_client,
                &api_key,
                &task_id,
                &event_tx,
            )
            .await
            {
                Ok(url) => url,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "GENERATION_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            let _ = event_tx
                .send(VideoStreamEvent::Progress {
                    percent: 85,
                    message: "Downloading video...".to_string(),
                })
                .await;

            // Download from temporary URL (expires in 24h)
            let video_bytes = match seedance::download_video(
                &gen_state.http_client,
                &video_url_temp,
            )
            .await
            {
                Ok(bytes) => bytes,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "DOWNLOAD_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            let _ = event_tx
                .send(VideoStreamEvent::Progress {
                    percent: 90,
                    message: "Uploading to storage...".to_string(),
                })
                .await;

            // Upload to S3
            let video_url = match s3_config
                .upload_bytes(video_bytes, &gen_user_id, "video/mp4", "mp4")
                .await
            {
                Ok(url) => url,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "UPLOAD_ERROR".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            // Debit credits
            let _ = billing::report_image_usage(
                &gen_state.http_client,
                &gen_state.z_billing_url,
                &gen_state.z_billing_api_key,
                &uuid::Uuid::new_v4().to_string(),
                &gen_user_id,
                provider,
                &model,
                cost_cents,
            )
            .await;

            // Store artifact
            if let Some(ref pid) = gen_project_id {
                if let (Some(ref surl), Some(ref stok)) =
                    (&gen_state.aura_storage_url, &gen_state.aura_storage_token)
                {
                    storage::store_artifact(
                        &gen_state.http_client,
                        surl,
                        stok,
                        pid,
                        &gen_user_id,
                        "video",
                        &video_url,
                        None,
                        None,
                        gen_name.as_deref(),
                        Some(&prompt),
                        None,
                        &model,
                        provider,
                        false,
                        None,
                    )
                    .await;
                }
            }

            let meta = VideoMeta {
                model,
                prompt,
                duration_seconds,
                resolution,
                aspect_ratio,
                provider: provider.to_string(),
                created: chrono::Utc::now().timestamp(),
            };

            let _ = event_tx
                .send(VideoStreamEvent::Completed { video_url, meta })
                .await;
        });
    } else {
        // Veo path — unchanged from original implementation
        tokio::spawn(async move {
            let _ = event_tx
                .send(VideoStreamEvent::Start {
                    ts: chrono::Utc::now().to_rfc3339(),
                })
                .await;

            // Submit task to Veo
            let operation_name = match veo::create_task(
                &gen_state.http_client,
                &api_key,
                &prompt,
                &model,
                &aspect_ratio,
                duration_seconds,
                &resolution,
                generate_audio,
            )
            .await
            {
                Ok(name) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Progress {
                            percent: 5,
                            message: "Video generation started...".to_string(),
                        })
                        .await;
                    name
                }
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "SUBMIT_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            // Poll for completion (sends progress events to keep SSE alive)
            let video_uri = match veo::poll_operation(
                &gen_state.http_client,
                &api_key,
                &operation_name,
                &event_tx,
            )
            .await
            {
                Ok(uri) => uri,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "GENERATION_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            let _ = event_tx
                .send(VideoStreamEvent::Progress {
                    percent: 85,
                    message: "Downloading video...".to_string(),
                })
                .await;

            // Download the video from Veo
            let video_bytes = match veo::download_video(
                &gen_state.http_client,
                &api_key,
                &video_uri,
            )
            .await
            {
                Ok(bytes) => bytes,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "DOWNLOAD_FAILED".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            let _ = event_tx
                .send(VideoStreamEvent::Progress {
                    percent: 90,
                    message: "Uploading to storage...".to_string(),
                })
                .await;

            // Upload to S3
            let video_url = match s3_config
                .upload_bytes(video_bytes, &gen_user_id, "video/mp4", "mp4")
                .await
            {
                Ok(url) => url,
                Err(e) => {
                    let _ = event_tx
                        .send(VideoStreamEvent::Error {
                            code: "UPLOAD_ERROR".to_string(),
                            message: e,
                        })
                        .await;
                    return;
                }
            };

            // Debit credits
            let _ = billing::report_image_usage(
                &gen_state.http_client,
                &gen_state.z_billing_url,
                &gen_state.z_billing_api_key,
                &uuid::Uuid::new_v4().to_string(),
                &gen_user_id,
                "google",
                &model,
                cost_cents,
            )
            .await;

            // Store artifact
            if let Some(ref pid) = gen_project_id {
                if let (Some(ref surl), Some(ref stok)) =
                    (&gen_state.aura_storage_url, &gen_state.aura_storage_token)
                {
                    storage::store_artifact(
                        &gen_state.http_client,
                        surl,
                        stok,
                        pid,
                        &gen_user_id,
                        "video",
                        &video_url,
                        None,
                        None,
                        gen_name.as_deref(),
                        Some(&prompt),
                        None,
                        &model,
                        "google",
                        false,
                        None,
                    )
                    .await;
                }
            }

            let meta = VideoMeta {
                model,
                prompt,
                duration_seconds,
                resolution,
                aspect_ratio,
                provider: "google".to_string(),
                created: chrono::Utc::now().timestamp(),
            };

            let _ = event_tx
                .send(VideoStreamEvent::Completed { video_url, meta })
                .await;
        });
    }

    // Stream SSE events to client
    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let json = serde_json::to_string(&event).unwrap_or_default();
            let sse = format!("data: {json}\n\n");
            yield Ok::<_, std::convert::Infallible>(sse);
        }
    };

    Ok((
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream".to_string()),
            (axum::http::header::CACHE_CONTROL, "no-cache".to_string()),
            (
                axum::http::header::HeaderName::from_static("x-accel-buffering"),
                "no".to_string(),
            ),
        ],
        axum::body::Body::from_stream(stream),
    )
        .into_response())
}

/// GET /v1/generate-video/config — Available models from all providers.
pub async fn generate_video_config(
    _auth: AuthUser,
) -> Json<VideoGenConfig> {
    let mut config = veo::get_config();
    config.models.extend(seedance::get_config());
    Json(config)
}
