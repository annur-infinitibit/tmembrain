//! `Error` invariants and conversion tests.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use membrain_core::error::{Error, Result};
use membrain_core::types::MemoryId;

fn requires_send_sync<T: Send + Sync>() {}

#[test]
fn error_is_send_and_sync() {
    requires_send_sync::<Error>();
}

#[test]
fn error_display_contains_metadata() {
    let err = Error::BudgetExceeded {
        memory_type: "episodic".to_string(),
        current: 10,
        max: 5,
    };
    let text = err.to_string();
    assert!(text.contains("episodic"));
    assert!(text.contains("10"));
    assert!(text.contains("5"));
}

#[test]
fn error_code_maps_every_major_variant() {
    let sample = [
        (Error::InvalidConfidence(0.5), "INVALID_CONFIDENCE"),
        (Error::MemoryNotFound(MemoryId::new()), "MEMORY_NOT_FOUND"),
        (Error::Storage("disk".into()), "STORAGE_ERROR"),
        (Error::Cancelled, "CANCELLED"),
        (Error::Timeout(1), "TIMEOUT"),
    ];
    for (error, code) in sample {
        assert_eq!(error.error_code(), code);
    }
}

#[test]
fn error_serde_json_conversion() {
    let serde_error = serde_json::from_str::<serde_json::Value>("not-json");
    assert!(serde_error.is_err());
    let converted: Error = serde_error.expect_err("err").into();
    assert_eq!(converted.error_code(), "SERIALIZATION_ERROR");
}

#[test]
fn error_toml_conversion() {
    let toml_error = toml::from_str::<serde_json::Value>("= invalid").expect_err("toml err");
    let converted: Error = toml_error.into();
    assert_eq!(converted.error_code(), "CONFIGURATION_ERROR");
}

#[test]
fn result_alias_usable_in_function_signature() {
    fn ok() -> Result<u8> {
        Ok(7)
    }
    assert_eq!(ok().expect("ok"), 7);
}

#[test]
fn is_retryable_covers_all_transient_variants() {
    assert!(Error::RateLimited {
        retry_after_secs: 1
    }
    .is_retryable());
    assert!(Error::DatabaseConnection("db down".into()).is_retryable());
    assert!(Error::Timeout(1).is_retryable());
    assert!(Error::WriteConflict(MemoryId::new()).is_retryable());
    assert!(!Error::Rejected {
        reason: "validation".into()
    }
    .is_retryable());
}
