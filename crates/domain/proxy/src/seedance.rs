//! BytePlus Seedance 2.0 video generation client.
//!
//! Submits text-to-video tasks via the ModelArk API and polls for completion.
//! Returns a download URL for the generated MP4.
//!
//! API reference: BytePlus ModelArk Video Generation API
//! Base URL: https://ark.ap-southeast.bytepluses.com/api/v3
//! Auth: Bearer token via ARK_API_KEY

use super::video_types::{VideoModelConfig, VideoStreamEvent};

const SEEDANCE_API_BASE: &str = "https://ark.ap-southeast.bytepluses.com/api/v3";
const POLL_INTERVAL_MS: u64 = 10_000; // 10s (from official SDK examples)
const MAX_POLL_ATTEMPTS: u32 = 90; // 90 × 10s = 15 min max
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Returns true if the model ID belongs to the Seedance provider.
pub fn is_seedance_model(model: &str) -> bool {
    model.starts_with("dreamina-seedance")
}

// ---------------------------------------------------------------------------
// Pixel dimensions for Seedance 2.0 (verified from official docs)
// These differ from Seedance 1.0 Pro dimensions.
// ---------------------------------------------------------------------------

/// Returns (width, height) for the given resolution and aspect ratio.
/// Dimensions are specific to Seedance 2.0 / 2.0 Fast models.
fn pixel_dimensions(resolution: &str, ratio: &str) -> (u32, u32) {
    match (resolution, ratio) {
        // 480p
        ("480p", "16:9") => (864, 496),
        ("480p", "9:16") => (496, 864),
        ("480p", "4:3") => (752, 560),
        ("480p", "3:4") => (560, 752),
        ("480p", "1:1") => (640, 640),
        ("480p", "21:9") => (992, 432),
        // 720p
        ("720p", "16:9") => (1280, 720),
        ("720p", "9:16") => (720, 1280),
        ("720p", "4:3") => (1112, 834),
        ("720p", "3:4") => (834, 1112),
        ("720p", "1:1") => (960, 960),
        ("720p", "21:9") => (1470, 630),
        // 1080p (standard model only)
        ("1080p", "16:9") => (1920, 1080),
        ("1080p", "9:16") => (1080, 1920),
        ("1080p", "4:3") => (1664, 1248),
        ("1080p", "3:4") => (1248, 1664),
        ("1080p", "1:1") => (1440, 1440),
        ("1080p", "21:9") => (2206, 946),
        // Fallback: 720p 16:9
        _ => (1280, 720),
    }
}

/// Estimate token consumption for a text-to-video generation.
///
/// Formula (from official docs):
///   tokens = duration × width × height × frame_rate / 1024
///
/// Frame rate is 24fps for all Seedance 2.0 models.
fn estimate_tokens(duration: u8, resolution: &str, ratio: &str) -> u64 {
    let (width, height) = pixel_dimensions(resolution, ratio);
    (duration as u64) * (width as u64) * (height as u64) * 24 / 1024
}

/// Calculate estimated cost in cents for a Seedance video generation.
///
/// Pricing is token-based with tiered rates by model and resolution.
/// A 20% markup is applied (same as Veo pricing).
///
/// Note: This is an estimate based on the official token formula.
/// Actual cost is determined by `usage.completion_tokens` in the API response.
pub fn cost_cents(model: &str, resolution: &str, ratio: &str, duration: u8) -> i64 {
    let tokens = estimate_tokens(duration, resolution, ratio);

    // Tiered pricing (USD per million tokens) — from official docs
    let price_per_million = match (model, resolution) {
        ("dreamina-seedance-2-0-260128", "1080p") => 7.7,
        ("dreamina-seedance-2-0-260128", _) => 7.0,
        ("dreamina-seedance-2-0-fast-260128", _) => 5.6,
        // Fallback for unknown Seedance models
        (_, "1080p") => 7.7,
        _ => 7.0,
    };

    let cost_usd = (tokens as f64) * price_per_million / 1_000_000.0;
    // 20% markup, convert to cents
    let total_usd = cost_usd * 1.2;
    (total_usd * 100.0).ceil() as i64
}

/// Submit a video generation task to the Seedance API.
/// Returns the task ID for polling.
pub async fn create_task(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    model: &str,
    aspect_ratio: &str,
    duration_seconds: u8,
    resolution: &str,
    generate_audio: bool,
) -> Result<String, String> {
    let url = format!("{SEEDANCE_API_BASE}/contents/generations/tasks");

    // Field name translations: our internal names → Seedance API names
    //   aspect_ratio → "ratio"
    //   duration_seconds → "duration"
    let body = serde_json::json!({
        "model": model,
        "content": [{
            "type": "text",
            "text": prompt
        }],
        "ratio": aspect_ratio,
        "duration": duration_seconds,
        "resolution": resolution,
        "generate_audio": generate_audio,
        "watermark": false
    });

    let resp = client
        .post(&url)
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Seedance request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Seedance error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let task_id = data["id"]
        .as_str()
        .ok_or("No task ID in Seedance response")?
        .to_string();

    Ok(task_id)
}

/// Poll a Seedance task until completion.
/// Returns the video download URL on success.
pub async fn poll_task(
    client: &reqwest::Client,
    api_key: &str,
    task_id: &str,
    event_tx: &tokio::sync::mpsc::Sender<VideoStreamEvent>,
) -> Result<String, String> {
    for attempt in 0..MAX_POLL_ATTEMPTS {
        tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;

        let percent = 10 + ((attempt as f32 / MAX_POLL_ATTEMPTS as f32) * 70.0) as u8;
        let _ = event_tx
            .send(VideoStreamEvent::Progress {
                percent,
                message: "Generating video...".to_string(),
            })
            .await;

        let url = format!("{SEEDANCE_API_BASE}/contents/generations/tasks/{task_id}");

        let resp = client
            .get(&url)
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .send()
            .await
            .map_err(|e| format!("Seedance poll failed: {e}"))?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_default();
            tracing::warn!(attempt, task_id, error = %error, "Seedance poll error, retrying");
            continue;
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

        let status = data["status"].as_str().unwrap_or("");

        match status {
            "succeeded" => {
                let video_url = data["content"]["video_url"]
                    .as_str()
                    .ok_or_else(|| {
                        tracing::error!(
                            task_id,
                            response = %data,
                            "Seedance succeeded but no video_url in response"
                        );
                        "No video_url in completed Seedance response".to_string()
                    })?
                    .to_string();

                return Ok(video_url);
            }
            "failed" => {
                let code = data["error"]["code"].as_str().unwrap_or("UNKNOWN");
                let message = data["error"]["message"]
                    .as_str()
                    .unwrap_or("Video generation failed");
                return Err(format!("Seedance error {code}: {message}"));
            }
            "expired" => {
                return Err(format!("Seedance task {task_id} expired"));
            }
            // "queued" | "running" | other → continue polling
            _ => {
                tracing::debug!(attempt, task_id, status, "Seedance task still processing");
                continue;
            }
        }
    }

    Err(format!(
        "Seedance task {task_id} timed out after {} seconds",
        MAX_POLL_ATTEMPTS as u64 * POLL_INTERVAL_MS / 1000
    ))
}

/// Download a video from its temporary URL and return the bytes.
///
/// Seedance video URLs are public TOS URLs — no auth header needed.
/// URLs expire after 24 hours, so we download immediately and re-upload to S3.
pub async fn download_video(client: &reqwest::Client, video_url: &str) -> Result<Vec<u8>, String> {
    let resp = client
        .get(video_url)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Video download failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Video download returned {}", resp.status()));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read video bytes: {e}"))?;

    if bytes.len() > 200 * 1024 * 1024 {
        return Err(format!("Video too large: {}MB", bytes.len() / 1024 / 1024));
    }

    Ok(bytes.to_vec())
}

/// Returns available Seedance model configurations.
pub fn get_config() -> Vec<VideoModelConfig> {
    vec![
        VideoModelConfig {
            id: "dreamina-seedance-2-0-260128".to_string(),
            name: "Seedance 2.0".to_string(),
            provider: "byteplus".to_string(),
            max_duration: 15,
            supports_audio: true,
            supports_4k: false,
        },
        VideoModelConfig {
            id: "dreamina-seedance-2-0-fast-260128".to_string(),
            name: "Seedance 2.0 Fast".to_string(),
            provider: "byteplus".to_string(),
            max_duration: 15,
            supports_audio: true,
            supports_4k: false,
        },
    ]
}
