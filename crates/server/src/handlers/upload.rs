//! Upload presigning handler — generates presigned S3 PUT URLs for
//! direct client-side uploads.

use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use aura_router_auth::AuthUser;
use aura_router_core::AppError;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct PresignRequest {
    pub content_type: String,
    pub filename: String,
}

/// POST /v1/upload/presign
///
/// Generates a presigned S3 PUT URL that the client can use to upload
/// a file directly to S3. Requires an authenticated user.
pub async fn presign_upload(
    auth: AuthUser,
    State(state): State<AppState>,
    Json(input): Json<PresignRequest>,
) -> Result<Json<aura_router_proxy::s3::PresignedUpload>, AppError> {
    let s3 = state
        .s3_config
        .as_ref()
        .ok_or_else(|| AppError::Internal("S3 not configured".into()))?;

    if input.filename.trim().is_empty() || input.filename.len() > 255 {
        return Err(AppError::BadRequest("Invalid filename".into()));
    }

    let result = s3
        .presign_upload(&auth.user_id, &input.content_type, &input.filename)
        .await
        .map_err(|e| AppError::BadRequest(e))?;

    Ok(Json(result))
}
