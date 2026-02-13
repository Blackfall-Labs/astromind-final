//! Genome-to-development parameter mapping.
//!
//! Maps a GenomeProfile (normalized 0.0-1.0 priors) to concrete modulation
//! factors that affect population mix, wiring config, and differentiation.

/// Normalized genome profile for development modulation.
///
/// Each field is 0.0 to 1.0. These map from MachineGenome priors but are
/// normalized and focused on the subset relevant to tissue development.
#[derive(Debug, Clone)]
pub struct GenomeProfile {
    /// NOV prior — novelty seeking. Higher = wider wiring, lower sensory thresholds.
    pub novelty: f32,
    /// GOV prior — governance/inhibition. Higher = more gate neurons, stronger inhibition.
    pub governance: f32,
    /// PLA prior — plasticity. Higher = more synaptic change capacity.
    pub plasticity: f32,
    /// PRE prior — precision. Higher = tighter wiring, higher thresholds.
    pub precision: f32,
    /// EXP prior — expression. Higher = more motor neurons, stronger output.
    pub expression: f32,
    /// DEV prior — developmental budget. Higher = larger regions.
    pub development: f32,
}

impl GenomeProfile {
    /// Create from raw 0.0-1.0 values.
    pub fn new(
        novelty: f32,
        governance: f32,
        plasticity: f32,
        precision: f32,
        expression: f32,
        development: f32,
    ) -> Self {
        Self {
            novelty: novelty.clamp(0.0, 1.0),
            governance: governance.clamp(0.0, 1.0),
            plasticity: plasticity.clamp(0.0, 1.0),
            precision: precision.clamp(0.0, 1.0),
            expression: expression.clamp(0.0, 1.0),
            development: development.clamp(0.0, 1.0),
        }
    }

    /// Neutral genome — 0.5 across all priors.
    pub fn neutral() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    }

    /// DEV factor for metabolic budget (0.5x to 2.0x).
    pub fn dev_factor(&self) -> f32 {
        0.5 + self.development * 1.5
    }

    /// Wiring reach modifier. High novelty = wider, high precision = tighter.
    pub fn wiring_reach_factor(&self) -> f32 {
        let base = 1.0;
        let novelty_bonus = (self.novelty - 0.5) * 0.6; // +-0.3
        let precision_penalty = (self.precision - 0.5) * -0.4; // +-0.2
        (base + novelty_bonus + precision_penalty).max(0.3)
    }

    /// Wiring fanout modifier. High novelty = more connections.
    pub fn wiring_fanout_factor(&self) -> f32 {
        let base = 1.0;
        let novelty_bonus = (self.novelty - 0.5) * 0.8;
        (base + novelty_bonus).max(0.3)
    }

    /// Gate neuron budget modifier. High governance = more gates.
    pub fn gate_fraction_modifier(&self) -> f32 {
        1.0 + (self.governance - 0.5) * 1.0 // 0.5x to 1.5x
    }

    /// Sensory neuron threshold modifier. High novelty = lower thresholds (more sensitive).
    pub fn sensory_threshold_factor(&self) -> f32 {
        1.0 - (self.novelty - 0.5) * 0.4 // 0.8x to 1.2x (inverted)
    }

    /// Motor neuron budget modifier. High expression = more motor neurons.
    pub fn motor_fraction_modifier(&self) -> f32 {
        1.0 + (self.expression - 0.5) * 0.8 // 0.6x to 1.4x
    }

    /// Differentiation pressure modifier. High plasticity = faster commitment.
    pub fn differentiation_rate(&self) -> f32 {
        1.0 + (self.plasticity - 0.5) * 0.6 // 0.7x to 1.3x
    }
}

impl Default for GenomeProfile {
    fn default() -> Self {
        Self::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_is_centered() {
        let g = GenomeProfile::neutral();
        assert!((g.dev_factor() - 1.25).abs() < 0.01);
        assert!((g.wiring_reach_factor() - 1.0).abs() < 0.01);
        assert!((g.gate_fraction_modifier() - 1.0).abs() < 0.01);
    }

    #[test]
    fn high_novelty_widens_wiring() {
        let high = GenomeProfile::new(0.9, 0.5, 0.5, 0.5, 0.5, 0.5);
        let low = GenomeProfile::new(0.1, 0.5, 0.5, 0.5, 0.5, 0.5);

        assert!(
            high.wiring_reach_factor() > low.wiring_reach_factor(),
            "high novelty reach ({}) should exceed low ({})",
            high.wiring_reach_factor(),
            low.wiring_reach_factor()
        );
    }

    #[test]
    fn high_governance_more_gates() {
        let high = GenomeProfile::new(0.5, 0.9, 0.5, 0.5, 0.5, 0.5);
        let low = GenomeProfile::new(0.5, 0.1, 0.5, 0.5, 0.5, 0.5);

        assert!(
            high.gate_fraction_modifier() > low.gate_fraction_modifier(),
            "high governance ({}) should produce more gates than low ({})",
            high.gate_fraction_modifier(),
            low.gate_fraction_modifier()
        );
    }

    #[test]
    fn dev_factor_scales_budget() {
        let high = GenomeProfile::new(0.5, 0.5, 0.5, 0.5, 0.5, 1.0);
        let low = GenomeProfile::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.0);

        assert!((high.dev_factor() - 2.0).abs() < 0.01);
        assert!((low.dev_factor() - 0.5).abs() < 0.01);
    }

    #[test]
    fn values_clamped() {
        let g = GenomeProfile::new(5.0, -1.0, 0.5, 0.5, 0.5, 0.5);
        assert_eq!(g.novelty, 1.0);
        assert_eq!(g.governance, 0.0);
    }
}
