//! Integration tests for `Config` file/env round-trips and validation.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::fs;

use membrain_core::config::{Config, StorageConfig};
use tempfile::tempdir;

#[test]
fn config_default_is_valid() {
    let config = Config::default();
    config.validate().expect("default config must be valid");
}

#[test]
fn config_toml_roundtrip_via_file() {
    let directory = tempdir().expect("create tempdir");
    let path = directory.path().join("membrain.toml");

    let original = Config::default();
    original.save(&path).expect("save config");

    let contents = fs::read_to_string(&path).expect("read config");
    assert!(contents.contains("[storage]"), "toml missing storage table");

    let reloaded = Config::from_file(&path).expect("load config");
    assert_eq!(original.storage.backend, reloaded.storage.backend);
    assert_eq!(original.embedding.provider, reloaded.embedding.provider);
}

#[test]
fn config_from_toml_roundtrip_via_string() {
    let original = Config::default();
    let serialized = toml::to_string_pretty(&original).expect("serialize");
    let reloaded = Config::from_toml(&serialized).expect("parse");
    assert_eq!(original.storage.backend, reloaded.storage.backend);
    assert_eq!(
        original.write.salience.threshold,
        reloaded.write.salience.threshold,
    );
}

#[test]
fn config_rejects_unknown_backend() {
    let mut config = Config::default();
    config.storage.backend = "cassandra".to_string();
    let error = config.validate().expect_err("should reject cassandra");
    assert!(error.to_string().contains("cassandra"));
}

#[test]
fn config_rejects_out_of_range_salience_threshold() {
    let mut config = Config::default();
    config.write.salience.threshold = 1.5;
    let error = config.validate().expect_err("should reject high threshold");
    assert!(error.to_string().contains("salience"));
}

#[test]
fn config_rejects_out_of_range_novelty_threshold() {
    let mut config = Config::default();
    config.write.novelty.threshold = -0.1;
    let error = config.validate().expect_err("should reject negative threshold");
    assert!(error.to_string().contains("novelty"));
}

#[test]
fn storage_config_constructors_set_backend_field() {
    assert_eq!(StorageConfig::memscaledb("/tmp/db").backend, "memscaledb");
    assert_eq!(StorageConfig::sqlite("/tmp/db.sqlite").backend, "sqlite");
    assert_eq!(StorageConfig::memory().backend, "memory");
    assert_eq!(
        StorageConfig::postgres("postgres://localhost/db").backend,
        "postgres"
    );
}

#[test]
fn config_from_env_picks_up_storage_backend() {
    // Use a guard to restore env state even on panic.
    struct EnvGuard {
        keys: Vec<&'static str>,
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for key in &self.keys {
                // SAFETY: Only modifying env vars the test just set; no other
                // tests in this file mutate these.
                unsafe {
                    std::env::remove_var(key);
                }
            }
        }
    }

    let guard = EnvGuard {
        keys: vec![
            "MEMBRAIN_STORAGE_BACKEND",
            "MEMBRAIN_EMBEDDING_PROVIDER",
        ],
    };

    // SAFETY: same justification as the Drop impl.
    unsafe {
        std::env::set_var("MEMBRAIN_STORAGE_BACKEND", "sqlite");
        std::env::set_var("MEMBRAIN_EMBEDDING_PROVIDER", "openai");
    }
    let config = Config::from_env().expect("from_env");
    assert_eq!(config.storage.backend, "sqlite");
    assert_eq!(config.embedding.provider, "openai");
    drop(guard);
}

#[test]
fn config_builder_chain_produces_valid_config() {
    let config = Config::builder()
        .storage(StorageConfig::memory())
        .build()
        .expect("builder build");
    assert_eq!(config.storage.backend, "memory");
}
