//! Property-based tests for core types (Confidence, Embedding, AgentId, MemoryId).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

use std::time::Duration;

use membrain_core::types::{AgentId, Confidence, Embedding, MemoryId};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn confidence_always_in_range(value in any::<f64>()) {
        let confidence = Confidence::new(value);
        prop_assert!((0.0..=1.0).contains(&confidence.value()));
    }

    #[test]
    fn confidence_decay_exponential_never_increases(
        start in 0.0f64..=1.0,
        elapsed_secs in 0u64..=10_000,
        half_life_secs in 1u64..=10_000,
    ) {
        let before = Confidence::new(start);
        let after = before.decay_exponential(
            Duration::from_secs(elapsed_secs),
            Duration::from_secs(half_life_secs),
        );
        prop_assert!(after.value() <= before.value() + f64::EPSILON);
    }

    #[test]
    fn confidence_reinforce_never_decreases(
        start in 0.0f64..=1.0,
        factor in 0.0f64..=1.0,
    ) {
        let before = Confidence::new(start);
        let after = before.reinforce(factor);
        prop_assert!(after.value() + f64::EPSILON >= before.value());
    }

    #[test]
    fn confidence_weaken_never_increases(
        start in 0.0f64..=1.0,
        factor in 0.0f64..=1.0,
    ) {
        let before = Confidence::new(start);
        let after = before.weaken(factor);
        prop_assert!(after.value() <= before.value() + f64::EPSILON);
    }

    #[test]
    fn confidence_combine_and_commutative(a in 0.0f64..=1.0, b in 0.0f64..=1.0) {
        let left = Confidence::new(a).combine_and(&Confidence::new(b));
        let right = Confidence::new(b).combine_and(&Confidence::new(a));
        prop_assert!((left.value() - right.value()).abs() < 1e-9);
    }

    #[test]
    fn confidence_combine_or_commutative(a in 0.0f64..=1.0, b in 0.0f64..=1.0) {
        let left = Confidence::new(a).combine_or(&Confidence::new(b));
        let right = Confidence::new(b).combine_or(&Confidence::new(a));
        prop_assert!((left.value() - right.value()).abs() < 1e-9);
    }

    #[test]
    fn embedding_bytes_roundtrip(values in prop::collection::vec(-1000.0f32..=1000.0, 1..=128)) {
        let embedding = Embedding::new(values.clone());
        let bytes = embedding.to_bytes();
        let reparsed = Embedding::from_bytes(&bytes).expect("from_bytes");
        prop_assert_eq!(reparsed.values(), values.as_slice());
    }

    #[test]
    fn embedding_normalize_unit_length(values in prop::collection::vec(-10.0f32..=10.0, 2..=64)) {
        prop_assume!(values.iter().any(|v| v.abs() > 1e-3));
        let embedding = Embedding::new(values);
        let normalized = embedding.normalize();
        prop_assert!((normalized.norm() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn embedding_euclidean_symmetric(
        left_values in prop::collection::vec(-10.0f32..=10.0, 4..=32),
    ) {
        let right_values: Vec<f32> = left_values.iter().rev().copied().collect();
        let left = Embedding::new(left_values);
        let right = Embedding::new(right_values);
        let a = left.euclidean_distance(&right).expect("a->b");
        let b = right.euclidean_distance(&left).expect("b->a");
        prop_assert!((a - b).abs() < 1e-4);
    }

    #[test]
    fn embedding_cosine_self_is_one(values in prop::collection::vec(-10.0f32..=10.0, 4..=32)) {
        prop_assume!(values.iter().any(|v| v.abs() > 1e-3));
        let embedding = Embedding::new(values);
        let cosine = embedding.cosine_similarity(&embedding).expect("cosine self");
        prop_assert!((cosine - 1.0).abs() < 1e-3);
    }

    #[test]
    fn memory_id_bytes_roundtrip(bytes in any::<[u8; 16]>()) {
        let id = MemoryId::from_bytes(bytes);
        prop_assert_eq!(id.as_bytes(), &bytes);
    }

    #[test]
    fn agent_id_bytes_roundtrip(bytes in any::<[u8; 16]>()) {
        let id = AgentId::from_bytes(bytes);
        prop_assert_eq!(id.as_bytes(), &bytes);
    }

    #[test]
    fn agent_id_display_parses_as_uuid(bytes in any::<[u8; 16]>()) {
        let id = AgentId::from_bytes(bytes);
        let rendered = id.to_string();
        let parsed: uuid::Uuid = rendered.parse().expect("uuid parse");
        prop_assert_eq!(parsed.as_bytes(), &bytes);
    }
}

#[test]
fn confidence_constants_are_well_ordered() {
    assert!(Confidence::MIN.value() <= Confidence::LOW.value());
    assert!(Confidence::LOW.value() <= Confidence::DEFAULT.value());
    assert!(Confidence::DEFAULT.value() <= Confidence::HIGH.value());
    assert!(Confidence::HIGH.value() <= Confidence::MAX.value());
}

#[test]
fn confidence_try_new_rejects_out_of_range() {
    assert!(Confidence::try_new(-0.1).is_err());
    assert!(Confidence::try_new(1.1).is_err());
    assert!(Confidence::try_new(0.5).is_ok());
}

#[test]
fn embedding_from_bytes_rejects_odd_length() {
    let bytes = vec![0_u8; 13];
    assert!(Embedding::from_bytes(&bytes).is_err());
}

#[test]
fn embedding_with_dimension_rejects_mismatch() {
    let result = Embedding::with_dimension(vec![0.0; 4], 8);
    assert!(result.is_err());
}
