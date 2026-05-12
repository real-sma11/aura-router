//! Google Veo video generation client.
//!
//! Submits text-to-video tasks via the Gemini API and polls for completion.
//! Returns a download URL for the generated MP4.

use serde::{Deserialize, Serialize};

const VEO_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
const POLL_INTERVAL_MS: u64 = 10_000;
const MAX_POLL_ATTEMPTS: u32 = 40; // 40 × 10s = ~6.5 minutes max
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Video generation request from the client.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateVideoRequest {
    pub prompt: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_aspect_ratio")]
    pub aspect_ratio: String,
    #[serde(default = "default_duration")]
    pub duration_seconds: u8,
    #[serde(default = "default_resolution")]
    pub resolution: String,
    #[serde(default = "default_generate_audio")]
    pub generate_audio: bool,
    pub project_id: Option<String>,
    pub name: Option<String>,
}

fn default_model() -> String {
    "veo-3.1-fast-generate-preview".to_string()
}
fn default_aspect_ratio() -> String {
    "16:9".to_string()
}
fn default_duration() -> u8 {
    8
}
fn default_resolution() -> String {
    "720p".to_string()
}
fn default_generate_audio() -> bool {
    true
}

/// Response returned to the client after video generation completes.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateVideoResponse {
    pub success: bool,
    pub video_url: String,
    pub meta: VideoMeta,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoMeta {
    pub model: String,
    pub prompt: String,
    pub duration_seconds: u8,
    pub resolution: String,
    pub aspect_ratio: String,
    pub provider: String,
    pub created: i64,
}

/// SSE events emitted during video generation.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoStreamEvent {
    #[serde(rename = "start")]
    Start { ts: String },
    #[serde(rename = "progress")]
    Progress { percent: u8, message: String },
    #[serde(rename = "completed")]
    Completed {
        video_url: String,
        meta: VideoMeta,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

/// Available video model configurations.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoModelConfig {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub max_duration: u8,
    pub supports_audio: bool,
    pub supports_4k: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoGenConfig {
    pub default_model: String,
    pub models: Vec<VideoModelConfig>,
}

pub fn get_config() -> VideoGenConfig {
    VideoGenConfig {
        default_model: "veo-3.1-fast-generate-preview".to_string(),
        models: vec![
            VideoModelConfig {
                id: "veo-3.1-generate-preview".to_string(),
                name: "Veo 3.1 Standard".to_string(),
                provider: "google".to_string(),
                max_duration: 8,
                supports_audio: true,
                supports_4k: true,
            },
            VideoModelConfig {
                id: "veo-3.1-fast-generate-preview".to_string(),
                name: "Veo 3.1 Fast".to_string(),
                provider: "google".to_string(),
                max_duration: 8,
                supports_audio: true,
                supports_4k: true,
            },
            VideoModelConfig {
                id: "veo-3.1-lite-generate-preview".to_string(),
                name: "Veo 3.1 Lite".to_string(),
                provider: "google".to_string(),
                max_duration: 8,
                supports_audio: false,
                supports_4k: false,
            },
        ],
    }
}

/// Calculate cost in cents for a video generation based on model, resolution, and duration.
pub fn cost_cents(model: &str, resolution: &str, duration_seconds: u8) -> i64 {
    let per_second_usd = match (model, resolution) {
        ("veo-3.1-generate-preview", "4k") => 0.60,
        ("veo-3.1-generate-preview", _) => 0.40,
        ("veo-3.1-fast-generate-preview", "4k") => 0.30,
        ("veo-3.1-fast-generate-preview", "1080p") => 0.12,
        ("veo-3.1-fast-generate-preview", _) => 0.10,
        ("veo-3.1-lite-generate-preview", "1080p") => 0.08,
        ("veo-3.1-lite-generate-preview", _) => 0.05,
        (_, "4k") => 0.30,
        (_, "1080p") => 0.12,
        _ => 0.10,
    };
    // 20% markup, convert to cents
    let total_usd = per_second_usd * duration_seconds as f64 * 1.2;
    (total_usd * 100.0).ceil() as i64
}

/// Submit a video generation task to the Veo API.
/// Returns the operation name for polling.
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
    let url = format!(
        "{VEO_API_BASE}/models/{model}:predictLongRunning"
    );

    let body = serde_json::json!({
        "instances": [{
            "prompt": prompt
        }],
        "parameters": {
            "aspectRatio": aspect_ratio,
            "durationSeconds": duration_seconds,
            "resolution": resolution,
            "generateAudio": generate_audio,
            "personGeneration": "allow_adult"
        }
    });

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Veo request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Veo error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let operation_name = data["name"]
        .as_str()
        .ok_or("No operation name in Veo response")?
        .to_string();

    Ok(operation_name)
}

/// Poll a Veo operation until completion.
/// Returns the video download URI on success.
pub async fn poll_operation(
    client: &reqwest::Client,
    api_key: &str,
    operation_name: &str,
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

        let url = format!("{VEO_API_BASE}/{operation_name}");

        let resp = client
            .get(&url)
            .header("x-goog-api-key", api_key)
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .send()
            .await
            .map_err(|e| format!("Veo poll failed: {e}"))?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_default();
            tracing::warn!(attempt, operation_name, error = %error, "Veo poll error, retrying");
            continue;
        }

        let data: serde_json::Value =
            resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

        let done = data["done"].as_bool().unwrap_or(false);

        if !done {
            tracing::debug!(attempt, operation_name, "Veo operation still running");
            continue;
        }

        // Check for error in completed operation
        if let Some(error) = data.get("error") {
            let message = error["message"]
                .as_str()
                .unwrap_or("Video generation failed")
                .to_string();
            return Err(message);
        }

        // Extract video URI from completed response
        let video_uri = data
            .pointer("/response/generateVideoResponse/generatedSamples/0/video/uri")
            .and_then(|v| v.as_str())
            .ok_or("No video URI in completed Veo response")?
            .to_string();

        return Ok(video_uri);
    }

    Err(format!(
        "Veo operation {operation_name} timed out after {} seconds",
        MAX_POLL_ATTEMPTS as u64 * POLL_INTERVAL_MS / 1000
    ))
}

/// Download a video from the Veo API and return the bytes.
/// The download URL requires the API key as a query parameter.
pub async fn download_video(
    client: &reqwest::Client,
    api_key: &str,
    video_uri: &str,
) -> Result<Vec<u8>, String> {
    // Veo video URIs require the API key appended
    let download_url = if video_uri.contains('?') {
        format!("{video_uri}&key={api_key}")
    } else {
        format!("{video_uri}?key={api_key}")
    };

    let resp = client
        .get(&download_url)
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
        return Err(format!(
            "Video too large: {}MB",
            bytes.len() / 1024 / 1024
        ));
    }

    Ok(bytes.to_vec())
}
