use async_trait::async_trait;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;

use aura_router_core::AppError;

use crate::public_rate_limit::PublicRateLimiter;
use crate::validate::{TokenClaims, TokenValidator};

/// Authenticated user extracted from JWT in Authorization header.
/// Returns 401 if no valid token is present. Used by all endpoints
/// that require a real authenticated user (upload, billing, etc.).
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: String,
    pub claims: TokenClaims,
}

/// The synthetic user ID assigned to unauthenticated public requests.
pub const PUBLIC_GUEST_USER_ID: &str = "public-guest";

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync + AsRef<TokenValidator>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| AppError::Unauthorized("Missing authorization header".into()))?;

        let validator = state.as_ref();
        let claims = validator
            .validate(token)
            .await
            .map_err(AppError::Unauthorized)?;

        let user_id = claims
            .user_id()
            .ok_or_else(|| AppError::Unauthorized("Token missing user ID".into()))?
            .to_string();

        Ok(AuthUser { user_id, claims })
    }
}

/// Authenticated user OR unauthenticated public guest.
///
/// When a valid JWT is present, behaves identically to [`AuthUser`].
/// When no Authorization header is present, returns a public-guest
/// identity subject to IP-based rate limiting.
///
/// **Only use this extractor on endpoints that are intentionally
/// public.** All other endpoints should use [`AuthUser`] which
/// rejects unauthenticated requests with 401.
#[derive(Debug, Clone)]
pub struct AuthUserOrGuest {
    pub user_id: String,
    pub claims: TokenClaims,
}

impl AuthUserOrGuest {
    pub fn is_public_guest(&self) -> bool {
        self.user_id == PUBLIC_GUEST_USER_ID
    }

    /// Convert to `AuthUser` for passing to internal functions that
    /// expect the stricter type. Both types have identical fields.
    pub fn into_auth_user(self) -> AuthUser {
        AuthUser {
            user_id: self.user_id,
            claims: self.claims,
        }
    }
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthUserOrGuest
where
    S: Send + Sync + AsRef<TokenValidator> + AsRef<PublicRateLimiter>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        // No Authorization header → public-guest path with IP rate limiting.
        let Some(token) = token else {
            let ip = extract_client_ip(parts);
            let public_limiter: &PublicRateLimiter = state.as_ref();
            if let Err(retry_after) = public_limiter.check(&ip) {
                return Err(AppError::RateLimited {
                    retry_after,
                    message: "Public rate limit exceeded".into(),
                });
            }
            return Ok(AuthUserOrGuest {
                user_id: PUBLIC_GUEST_USER_ID.to_string(),
                claims: TokenClaims {
                    id: Some(PUBLIC_GUEST_USER_ID.to_string()),
                    sub: None,
                },
            });
        };

        let validator: &TokenValidator = state.as_ref();
        let claims = validator
            .validate(token)
            .await
            .map_err(AppError::Unauthorized)?;

        let user_id = claims
            .user_id()
            .ok_or_else(|| AppError::Unauthorized("Token missing user ID".into()))?
            .to_string();

        Ok(AuthUserOrGuest { user_id, claims })
    }
}

/// Extract the client IP from request headers. Checks X-Forwarded-For
/// (first hop) then X-Real-IP, falls back to 127.0.0.1.
fn extract_client_ip(parts: &Parts) -> String {
    if let Some(forwarded) = parts
        .headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return forwarded.to_string();
    }
    if let Some(real) = parts
        .headers
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return real.to_string();
    }
    "127.0.0.1".to_string()
}

/// Internal service auth extracted from X-Internal-Token header.
#[derive(Debug, Clone)]
pub struct InternalAuth;

/// Wrapper for the internal service token, stored in AppState.
#[derive(Clone)]
pub struct InternalToken(pub String);

#[async_trait]
impl<S> FromRequestParts<S> for InternalAuth
where
    S: Send + Sync + AsRef<InternalToken>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get("x-internal-token")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| AppError::Unauthorized("Missing internal token".into()))?;

        let expected = state.as_ref();
        if token != expected.0 {
            return Err(AppError::Unauthorized("Invalid internal token".into()));
        }

        Ok(InternalAuth)
    }
}
