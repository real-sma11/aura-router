//! IP-based rate limiter for unauthenticated public-guest requests.
//!
//! Separate from the per-user rate limiter in `aura-router-proxy` —
//! this one gates the `AuthUser` extractor itself so rate-limited
//! public requests never reach the proxy handler or incur an LLM call.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Hardcoded limits for public-guest requests. Tunable by changing
/// the constants and redeploying — no env vars needed.
const PUBLIC_MAX_REQUESTS_PER_WINDOW: u32 = 20;
const PUBLIC_WINDOW_SECS: u64 = 3600; // 1 hour

/// Per-IP rate limiter for public-guest requests.
pub struct PublicRateLimiter {
    state: Mutex<HashMap<String, (u32, Instant)>>,
}

impl PublicRateLimiter {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Check if a public request from this IP is allowed.
    /// Returns `Ok(())` if allowed, `Err(retry_after_secs)` if limited.
    pub fn check(&self, ip: &str) -> Result<(), u64> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let entry = state.entry(ip.to_string()).or_insert((0, now));

        // Reset window if expired
        if now.duration_since(entry.1).as_secs() >= PUBLIC_WINDOW_SECS {
            entry.0 = 0;
            entry.1 = now;
        }

        if entry.0 >= PUBLIC_MAX_REQUESTS_PER_WINDOW {
            let retry_after = PUBLIC_WINDOW_SECS - now.duration_since(entry.1).as_secs();
            return Err(retry_after);
        }

        entry.0 += 1;
        Ok(())
    }
}

impl Default for PublicRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}
