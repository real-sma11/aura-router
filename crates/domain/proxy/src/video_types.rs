//! Shared types for video generation providers (Veo, Seedance, etc.).
//!
//! These types are provider-agnostic and used by the video generation handler
//! and all provider modules.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request deserialization
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SSE event types
// ---------------------------------------------------------------------------

/// SSE events emitted during video generation.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoStreamEvent {
    #[serde(rename = "start")]
    Start { ts: String },
    #[serde(rename = "progress")]
    Progress { percent: u8, message: String },
    #[serde(rename = "completed")]
    Completed { video_url: String, meta: VideoMeta },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

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
