//! Confidence score with constrained range [0.0, 1.0] and decay methods

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

use crate::error::{Error, Result};

/// A confidence score constrained to the range [0.0, 1.0]
///
/// Represents certainty in a memory's accuracy or relevance.
/// Supports decay operations for memory consolidation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Confidence(f64);

impl Confidence {
    /// Minimum confidence value
    pub const MIN: Confidence = Confidence(0.0);

    /// Maximum confidence value
    pub const MAX: Confidence = Confidence(1.0);

    /// Default confidence for new memories
    pub const DEFAULT: Confidence = Confidence(0.5);

    /// High confidence threshold
    pub const HIGH: Confidence = Confidence(0.8);

    /// Low confidence threshold
    pub const LOW: Confidence = Confidence(0.3);

    /// Create a new confidence value, clamping to [0.0, 1.0]
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Try to create a confidence value, returning error if out of range
    pub fn try_new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::InvalidConfidence(value));
        }
        if value.is_nan() {
            return Err(Error::InvalidConfidence(value));
        }
        Ok(Self(value))
    }

    /// Get the underlying f64 value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if this is high confidence (>= 0.8)
    pub fn is_high(&self) -> bool {
        self.0 >= Self::HIGH.0
    }

    /// Check if this is low confidence (<= 0.3)
    pub fn is_low(&self) -> bool {
        self.0 <= Self::LOW.0
    }

    /// Apply exponential decay based on time elapsed
    ///
    /// Uses the formula: confidence * e^(-decay_rate * time_hours)
    pub fn decay_exponential(&self, elapsed: Duration, half_life: Duration) -> Self {
        if half_life.is_zero() {
            return Self::MIN;
        }

        // Calculate decay constant from half-life: λ = ln(2) / half_life
        let decay_constant = std::f64::consts::LN_2 / half_life.as_secs_f64();
        let elapsed_secs = elapsed.as_secs_f64();

        let decayed = self.0 * (-decay_constant * elapsed_secs).exp();
        Self::new(decayed)
    }

    /// Apply linear decay based on time elapsed
    ///
    /// Decreases by a fixed amount per unit time
    pub fn decay_linear(&self, elapsed: Duration, decay_per_day: f64) -> Self {
        let days = elapsed.as_secs_f64() / 86400.0;
        let decayed = self.0 - (decay_per_day * days);
        Self::new(decayed)
    }

    /// Reinforce confidence (increase it), approaching 1.0
    ///
    /// Uses: new = old + (1 - old) * factor
    pub fn reinforce(&self, factor: f64) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        let reinforced = self.0 + (1.0 - self.0) * factor;
        Self::new(reinforced)
    }

    /// Weaken confidence (decrease it), approaching 0.0
    ///
    /// Uses: new = old * (1 - factor)
    pub fn weaken(&self, factor: f64) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        let weakened = self.0 * (1.0 - factor);
        Self::new(weakened)
    }

    /// Combine two confidence scores using multiplication (AND semantics)
    pub fn combine_and(&self, other: &Confidence) -> Self {
        Self::new(self.0 * other.0)
    }

    /// Combine two confidence scores using noisy-or (OR semantics)
    /// P(A or B) = 1 - (1-A)(1-B)
    pub fn combine_or(&self, other: &Confidence) -> Self {
        Self::new(1.0 - (1.0 - self.0) * (1.0 - other.0))
    }

    /// Weighted average of multiple confidence scores
    pub fn weighted_average(scores: &[(Confidence, f64)]) -> Option<Self> {
        if scores.is_empty() {
            return None;
        }

        let total_weight: f64 = scores.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return None;
        }

        let weighted_sum: f64 = scores.iter().map(|(c, w)| c.0 * w).sum();
        Some(Self::new(weighted_sum / total_weight))
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

impl TryFrom<f64> for Confidence {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::try_new(value)
    }
}

impl From<Confidence> for f64 {
    fn from(c: Confidence) -> f64 {
        c.0
    }
}

impl std::ops::Mul<f64> for Confidence {
    type Output = Confidence;

    fn mul(self, rhs: f64) -> Self::Output {
        Confidence::new(self.0 * rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_clamping() {
        assert_eq!(Confidence::new(-0.5).value(), 0.0);
        assert_eq!(Confidence::new(1.5).value(), 1.0);
        assert_eq!(Confidence::new(0.5).value(), 0.5);
    }

    #[test]
    fn confidence_try_new() {
        assert!(Confidence::try_new(0.5).is_ok());
        assert!(Confidence::try_new(-0.1).is_err());
        assert!(Confidence::try_new(1.1).is_err());
        assert!(Confidence::try_new(f64::NAN).is_err());
    }

    #[test]
    fn confidence_exponential_decay() {
        let c = Confidence::new(1.0);
        let half_life = Duration::from_secs(3600); // 1 hour

        // After one half-life, should be ~0.5
        let decayed = c.decay_exponential(half_life, half_life);
        assert!((decayed.value() - 0.5).abs() < 0.01);

        // After two half-lives, should be ~0.25
        let decayed2 = c.decay_exponential(Duration::from_secs(7200), half_life);
        assert!((decayed2.value() - 0.25).abs() < 0.01);
    }

    #[test]
    fn confidence_linear_decay() {
        let c = Confidence::new(1.0);
        let decayed = c.decay_linear(Duration::from_secs(86400), 0.1); // 1 day, 0.1 per day
        assert!((decayed.value() - 0.9).abs() < 0.001);
    }

    #[test]
    fn confidence_reinforce() {
        let c = Confidence::new(0.5);
        let reinforced = c.reinforce(0.5);
        assert!((reinforced.value() - 0.75).abs() < 0.001);
    }

    #[test]
    fn confidence_weaken() {
        let c = Confidence::new(0.8);
        let weakened = c.weaken(0.5);
        assert!((weakened.value() - 0.4).abs() < 0.001);
    }

    #[test]
    fn confidence_combine_and() {
        let a = Confidence::new(0.8);
        let b = Confidence::new(0.5);
        let combined = a.combine_and(&b);
        assert!((combined.value() - 0.4).abs() < 0.001);
    }

    #[test]
    fn confidence_combine_or() {
        let a = Confidence::new(0.8);
        let b = Confidence::new(0.5);
        let combined = a.combine_or(&b);
        // 1 - (1-0.8)(1-0.5) = 1 - 0.2*0.5 = 0.9
        assert!((combined.value() - 0.9).abs() < 0.001);
    }

    #[test]
    fn confidence_weighted_average() {
        let scores = vec![(Confidence::new(0.8), 2.0), (Confidence::new(0.4), 1.0)];
        let avg = Confidence::weighted_average(&scores).unwrap();
        // (0.8*2 + 0.4*1) / 3 = 2.0 / 3 ≈ 0.667
        assert!((avg.value() - 0.667).abs() < 0.01);
    }

    #[test]
    fn confidence_serialization() {
        let c = Confidence::new(0.75);
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "0.75");

        let restored: Confidence = serde_json::from_str(&json).unwrap();
        assert_eq!(c, restored);
    }

    #[test]
    fn confidence_never_increases_on_decay() {
        let c = Confidence::new(0.8);
        let half_life = Duration::from_secs(3600);

        for hours in 0..100 {
            let elapsed = Duration::from_secs(hours * 3600);
            let decayed = c.decay_exponential(elapsed, half_life);
            assert!(decayed.value() <= c.value());
        }
    }
}
