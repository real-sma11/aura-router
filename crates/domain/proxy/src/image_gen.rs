//! Image generation clients for OpenAI and Gemini.

use base64::Engine;
use serde::{Deserialize, Serialize};

/// Image generation request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageRequest {
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    pub model: Option<String>,
    /// Caller-requested image quality. Validated per model in
    /// [`resolve_quality`]; ignored when absent or unsupported.
    pub quality: Option<String>,
    pub images: Option<Vec<ImageInput>>,
    pub prompt_mode: Option<String>,
    #[serde(default)]
    pub is_iteration: bool,
    pub project_id: Option<String>,
    pub parent_id: Option<String>,
    pub name: Option<String>,
}

fn default_size() -> String {
    "1024x1024".to_string()
}

/// Image input — either a URL or base64 data URL.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ImageInput {
    Url(String),
    Object { url: String, name: Option<String> },
}

impl ImageInput {
    pub fn url(&self) -> &str {
        match self {
            ImageInput::Url(u) => u,
            ImageInput::Object { url, .. } => url,
        }
    }
}

/// Image generation response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageResponse {
    pub success: bool,
    pub image_url: String,
    pub original_url: Option<String>,
    pub meta: ImageMeta,
}

/// Image generation metadata.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageMeta {
    pub model: String,
    pub size: String,
    pub prompt: String,
    pub provider: String,
    pub created: i64,
}

/// Config for available models.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenConfig {
    pub models: Vec<ImageModelConfig>,
    pub default_model: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModelConfig {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub eta_ms: u64,
    pub supports_references: bool,
}

pub fn get_config() -> ImageGenConfig {
    ImageGenConfig {
        default_model: "gpt-image-2".to_string(),
        models: vec![
            ImageModelConfig {
                id: "gpt-image-2".to_string(),
                name: "GPT Image 2".to_string(),
                provider: "openai".to_string(),
                eta_ms: 20000,
                supports_references: true,
            },
            ImageModelConfig {
                id: "gpt-image-1".to_string(),
                name: "GPT Image 1".to_string(),
                provider: "openai".to_string(),
                eta_ms: 20000,
                supports_references: true,
            },
            ImageModelConfig {
                id: "dall-e-3".to_string(),
                name: "DALL-E 3".to_string(),
                provider: "openai".to_string(),
                eta_ms: 15000,
                supports_references: false,
            },
            ImageModelConfig {
                id: "gemini-nano-banana".to_string(),
                name: "Gemini Flash Image".to_string(),
                provider: "google".to_string(),
                eta_ms: 25000,
                supports_references: true,
            },
        ],
    }
}

/// Resolve the OpenAI `quality` parameter for a model, honoring a
/// caller-supplied preference when it is valid for that model.
///
/// GPT Image models accept `low` / `medium` / `high` / `auto` (defaulting
/// to `high` for backward compatibility when nothing valid is requested).
/// DALL-E 3 accepts `standard` / `hd` (defaulting to `hd`). Other models
/// have no quality parameter.
fn resolve_quality(model: &str, requested: Option<&str>) -> Option<String> {
    let normalized = requested.map(|s| s.trim().to_ascii_lowercase());
    match model {
        "gpt-image-1" | "gpt-image-2" => Some(
            normalized
                .filter(|q| matches!(q.as_str(), "low" | "medium" | "high" | "auto"))
                .unwrap_or_else(|| "high".to_string()),
        ),
        "dall-e-3" => Some(
            normalized
                .filter(|q| matches!(q.as_str(), "standard" | "hd"))
                .unwrap_or_else(|| "hd".to_string()),
        ),
        _ => None,
    }
}

/// Result from image generation — base64 image data.
pub struct GeneratedImage {
    pub base64_data: String,
    pub mime_type: String,
    pub model: String,
    pub provider: String,
}

/// Generate an image using OpenAI's API.
pub async fn generate_openai(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    quality: Option<&str>,
    images: Option<&[ImageInput]>,
    _is_iteration: bool,
) -> Result<GeneratedImage, String> {
    let has_images = images.map_or(false, |imgs| !imgs.is_empty());

    if has_images {
        generate_openai_edit(client, api_key, prompt, size, model, quality, images.unwrap()).await
    } else {
        generate_openai_create(client, api_key, prompt, size, model, quality).await
    }
}

/// OpenAI image generation (no reference images).
async fn generate_openai_create(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    quality: Option<&str>,
) -> Result<GeneratedImage, String> {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1
    });
    if let Some(q) = resolve_quality(model, quality) {
        body["quality"] = serde_json::Value::String(q);
    }

    let resp = client
        .post("https://api.openai.com/v1/images/generations")
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("OpenAI request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let b64 = data["data"][0]["b64_json"]
        .as_str()
        .ok_or("No image data in response")?
        .to_string();

    Ok(GeneratedImage {
        base64_data: b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// OpenAI image edit (with reference images).
async fn generate_openai_edit(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    quality: Option<&str>,
    images: &[ImageInput],
) -> Result<GeneratedImage, String> {
    // Fetch first reference image as bytes
    let image_url = images[0].url();
    let image_bytes = fetch_image_bytes(client, image_url).await?;

    // Build multipart form
    let mut form = reqwest::multipart::Form::new()
        .text("model", model.to_string())
        .text("prompt", prompt.to_string())
        .text("size", size.to_string())
        .part(
            "image",
            reqwest::multipart::Part::bytes(image_bytes)
                .file_name("image.png")
                .mime_str("image/png")
                .map_err(|e| format!("MIME error: {e}"))?,
        );
    if let Some(q) = resolve_quality(model, quality) {
        form = form.text("quality", q);
    }

    let resp = client
        .post("https://api.openai.com/v1/images/edits")
        .header("authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("OpenAI edit request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        // Fallback to generation without references
        tracing::warn!("OpenAI edit failed, falling back to generation: {error}");
        return generate_openai_create(client, api_key, prompt, size, model, quality).await;
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    let b64 = data["data"][0]["b64_json"]
        .as_str()
        .ok_or("No image data in response")?
        .to_string();

    Ok(GeneratedImage {
        base64_data: b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// Generate an image using Google Gemini.
pub async fn generate_gemini(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    images: Option<&[ImageInput]>,
    _is_iteration: bool,
) -> Result<GeneratedImage, String> {
    // Add size instruction
    let size_instruction = match size {
        "1024x1024" | "256x256" | "512x512" => " Generate a square image (1:1 aspect ratio).",
        "1536x1024" => " Generate a landscape image (3:2 aspect ratio).",
        "1024x1536" => " Generate a portrait image (2:3 aspect ratio).",
        _ => "",
    };

    let prompt_with_size = format!("{prompt}{size_instruction}");

    // Build content parts
    let mut parts = Vec::new();

    // Add reference images if provided
    if let Some(imgs) = images {
        for img in imgs {
            if let Ok((data, mime)) = fetch_image_as_base64(client, img.url()).await {
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": data
                    }
                }));
            }
        }
    }

    // Add text prompt
    parts.push(serde_json::json!({ "text": prompt_with_size }));

    let body = serde_json::json!({
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    });

    let url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent";

    let resp = client
        .post(url)
        .header("content-type", "application/json")
        .header("x-goog-api-key", api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Gemini request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini error: {error}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| format!("Parse error: {e}"))?;

    // Extract image from response
    let parts = data
        .pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .ok_or("No parts in Gemini response")?;

    for part in parts {
        if let Some(inline_data) = part.get("inlineData") {
            let b64 = inline_data["data"]
                .as_str()
                .ok_or("No data in inlineData")?
                .to_string();
            let mime = inline_data["mimeType"]
                .as_str()
                .unwrap_or("image/png")
                .to_string();

            return Ok(GeneratedImage {
                base64_data: b64,
                mime_type: mime,
                model: "gemini-2.5-flash-image".to_string(),
                provider: "google".to_string(),
            });
        }
    }

    Err("No image found in Gemini response".to_string())
}

/// Check if an IP address is private/internal (loopback, private, link-local).
fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()          // 127.0.0.0/8
            || v4.is_private()        // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            || v4.is_link_local()     // 169.254.0.0/16 (cloud metadata)
            || v4.is_unspecified()    // 0.0.0.0
            || v4.is_broadcast()      // 255.255.255.255
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()          // ::1
            || v6.is_unspecified()    // ::
            // IPv4-mapped IPv6 (::ffff:x.x.x.x)
            || v6.to_ipv4_mapped().map_or(false, |v4| {
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
            })
        }
    }
}

/// Validate that a URL is safe to fetch (no SSRF).
fn validate_fetch_url(url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("Invalid URL: {e}"))?;

    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("Unsupported URL scheme: {scheme}")),
    }

    let host = parsed.host_str().ok_or("URL has no host")?;

    // Try parsing as IP address first (handles decimal, hex, octal, IPv6 forms)
    // reqwest::Url normalises IPs during parsing, so host_str() for an IP URL
    // will be the canonical form (e.g. "127.0.0.1" for decimal 2130706433).
    // Also try stripping brackets for IPv6.
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        if is_private_ip(ip) {
            return Err("URL points to a private/internal address".into());
        }
    }

    // Block known internal hostnames
    if host == "localhost"
        || host == "metadata.google.internal"
        || host.ends_with(".internal")
        || host.ends_with(".local")
    {
        return Err("URL points to a private/internal address".into());
    }

    Ok(())
}

/// Build a reqwest client that does not follow redirects (SSRF protection).
fn no_redirect_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap_or_default()
}

/// Fetch an image URL as raw bytes.
async fn fetch_image_bytes(_client: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    if url.starts_with("data:") {
        // Base64 data URL
        let b64 = url
            .find(',')
            .map(|i| &url[i + 1..])
            .unwrap_or(url);
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("Invalid base64: {e}"));
    }

    validate_fetch_url(url)?;

    let resp = no_redirect_client()
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch image: {e}"))?;

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Failed to read image: {e}"))
}

/// Fetch an image URL as base64 data + MIME type.
async fn fetch_image_as_base64(
    _client: &reqwest::Client,
    url: &str,
) -> Result<(String, String), String> {
    if url.starts_with("data:") {
        // Extract MIME and base64 from data URL
        let mime = url
            .strip_prefix("data:")
            .and_then(|s| s.split(';').next())
            .unwrap_or("image/png")
            .to_string();
        let b64 = url.find(',').map(|i| &url[i + 1..]).unwrap_or(url);
        return Ok((b64.to_string(), mime));
    }

    validate_fetch_url(url)?;

    let resp = no_redirect_client()
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch image: {e}"))?;

    let mime = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("image/png")
        .to_string();

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image: {e}"))?;

    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok((b64, mime))
}

/// SSE event types for streaming image generation.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageStreamEvent {
    #[serde(rename = "start")]
    Start { ts: String },
    #[serde(rename = "progress")]
    Progress { percent: u32, message: String },
    #[serde(rename = "partial-image")]
    PartialImage { data: String },
    #[serde(rename = "completed")]
    Completed {
        #[serde(rename = "imageUrl")]
        image_url: String,
        #[serde(rename = "originalUrl")]
        original_url: Option<String>,
        #[serde(rename = "artifactId", skip_serializing_if = "Option::is_none")]
        artifact_id: Option<String>,
        meta: ImageMeta,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

/// Generate an image using OpenAI with streaming (partial images).
pub async fn generate_openai_stream(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
    size: &str,
    model: &str,
    quality: Option<&str>,
    images: Option<&[ImageInput]>,
    _is_iteration: bool,
    event_tx: tokio::sync::mpsc::Sender<ImageStreamEvent>,
) -> Result<GeneratedImage, String> {
    let has_images = images.map_or(false, |imgs| !imgs.is_empty());

    // Send start event
    let _ = event_tx
        .send(ImageStreamEvent::Start {
            ts: chrono::Utc::now().to_rfc3339(),
        })
        .await;

    let _ = event_tx
        .send(ImageStreamEvent::Progress {
            percent: 10,
            message: "Generating image...".to_string(),
        })
        .await;

    // For streaming, OpenAI supports stream=true on images.generate
    // but the edit endpoint doesn't support streaming well.
    // Use non-streaming for edits, streaming for generation.
    if has_images {
        let _ = event_tx
            .send(ImageStreamEvent::Progress {
                percent: 30,
                message: "Processing reference images...".to_string(),
            })
            .await;

        let result =
            generate_openai_edit(client, api_key, prompt, size, model, quality, images.unwrap())
                .await?;

        let _ = event_tx
            .send(ImageStreamEvent::Progress {
                percent: 90,
                message: "Uploading...".to_string(),
            })
            .await;

        return Ok(result);
    }

    // Try streaming generation
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "stream": true,
        "partial_images": 2
    });
    if let Some(q) = resolve_quality(model, quality) {
        body["quality"] = serde_json::Value::String(q);
    }

    let resp = client
        .post("https://api.openai.com/v1/images/generations")
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("OpenAI streaming request failed: {e}"))?;

    if !resp.status().is_success() {
        let error = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI error: {error}"));
    }

    // Parse SSE stream from OpenAI
    let mut final_b64 = String::new();
    let mut stream = resp.bytes_stream();

    use futures_util::StreamExt;
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Process complete SSE lines
        while let Some(newline_pos) = buffer.find("\n\n") {
            let event_block = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 2..].to_string();

            // Parse event
            let mut event_type = String::new();
            let mut data = String::new();

            for line in event_block.lines() {
                if let Some(et) = line.strip_prefix("event: ") {
                    event_type = et.to_string();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = d.to_string();
                }
            }

            if data == "[DONE]" {
                break;
            }

            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&data) {
                match event_type.as_str() {
                    "image_generation.partial_image" | "image_edit.partial_image" => {
                        if let Some(b64) = value.get("b64_json").and_then(|v| v.as_str()) {
                            let _ = event_tx
                                .send(ImageStreamEvent::PartialImage {
                                    data: format!("data:image/png;base64,{b64}"),
                                })
                                .await;
                        }

                        let _ = event_tx
                            .send(ImageStreamEvent::Progress {
                                percent: 50,
                                message: "Refining...".to_string(),
                            })
                            .await;
                    }
                    "image_generation.completed" | "image_edit.completed" => {
                        if let Some(b64) = value.get("b64_json").and_then(|v| v.as_str()) {
                            final_b64 = b64.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if final_b64.is_empty() {
        return Err("No final image in stream".to_string());
    }

    let _ = event_tx
        .send(ImageStreamEvent::Progress {
            percent: 90,
            message: "Uploading...".to_string(),
        })
        .await;

    Ok(GeneratedImage {
        base64_data: final_b64,
        mime_type: "image/png".to_string(),
        model: model.to_string(),
        provider: "openai".to_string(),
    })
}

/// Resolve which provider/model to use.
/// If promptMode is set, it overrides explicit model selection (matching original AURA):
/// - "new" or "remix" → gpt-image-1
/// - "edit" → gemini-nano-banana
pub fn resolve_image_model(model: Option<&str>, prompt_mode: Option<&str>) -> (&'static str, &'static str) {
    // promptMode takes precedence if set
    if let Some(mode) = prompt_mode {
        return match mode {
            "new" | "remix" => ("gpt-image-1", "openai"),
            "edit" => ("gemini-nano-banana", "google"),
            _ => ("gpt-image-1", "openai"),
        };
    }

    match model {
        Some("gemini-nano-banana") | Some("gemini") => ("gemini-nano-banana", "google"),
        Some("dall-e-3") => ("dall-e-3", "openai"),
        Some("dall-e-2") => ("dall-e-2", "openai"),
        Some("gpt-image-1") => ("gpt-image-1", "openai"),
        Some("gpt-image-2") => ("gpt-image-2", "openai"),
        Some("gpt-4o") => ("gpt-4o", "openai"),
        _ => ("gpt-image-2", "openai"),
    }
}
