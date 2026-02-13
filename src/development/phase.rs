//! Development phase configuration.
//!
//! Controls how many rounds each developmental phase runs.
//! Phases are sequential: Exposure -> Differentiation -> Crystallization.

/// Configuration for development phase timing.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    /// Rounds of trophic field diffusion and pressure accumulation.
    pub exposure_rounds: usize,
    /// Rounds of neuron commitment and nuclei blending.
    pub differentiation_rounds: usize,
    /// Max neurons to commit per differentiation round.
    pub max_commits_per_round: usize,
    /// Nuclei blend factor per round (0.0 = no change, 1.0 = snap to target).
    pub blend_rate: f32,
    /// Pressure accumulation rate multiplier.
    pub pressure_rate: f32,
}

impl PhaseConfig {
    /// Default configuration suitable for most regions.
    pub fn default_region() -> Self {
        PhaseConfig {
            exposure_rounds: 500,
            differentiation_rounds: 200,
            max_commits_per_round: 50,
            blend_rate: 0.3,
            pressure_rate: 1.0,
        }
    }

    /// Fast configuration for testing.
    pub fn fast() -> Self {
        PhaseConfig {
            exposure_rounds: 50,
            differentiation_rounds: 30,
            max_commits_per_round: 100,
            blend_rate: 0.5,
            pressure_rate: 2.0,
        }
    }

    /// Modulate phase timing by genome plasticity.
    /// High plasticity = faster differentiation, more pressure.
    pub fn with_genome_modulation(mut self, plasticity_rate: f32) -> Self {
        self.pressure_rate *= plasticity_rate;
        self.blend_rate = (self.blend_rate * plasticity_rate).clamp(0.1, 0.8);
        self
    }
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self::default_region()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_reasonable() {
        let config = PhaseConfig::default_region();
        assert!(config.exposure_rounds > 0);
        assert!(config.differentiation_rounds > 0);
        assert!(config.blend_rate > 0.0 && config.blend_rate < 1.0);
    }

    #[test]
    fn fast_is_faster() {
        let fast = PhaseConfig::fast();
        let default = PhaseConfig::default_region();
        assert!(fast.exposure_rounds < default.exposure_rounds);
    }

    #[test]
    fn genome_modulation_affects_rate() {
        let base = PhaseConfig::default_region();
        let boosted = base.clone().with_genome_modulation(1.5);
        assert!(boosted.pressure_rate > base.pressure_rate);
    }
}
