//! Development executor — turns parsed .neuro programs into spatial neuron pools.
//!
//! Pipeline: scatter -> trophic fields -> disc programs -> exposure ->
//! differentiation -> wiring -> DevelopedRegion

pub mod differentiation;
pub mod executor;
pub mod phase;
pub mod scatter;

pub use executor::{DevelopedRegion, DevelopmentStats};
pub use phase::PhaseConfig;
