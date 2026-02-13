//! Metabolic budget — finite resource pool for region development.
//!
//! Each region has a metabolic envelope (total neuron budget) determined by
//! its proportion symbol and genome modulation. The budget constrains how
//! many neurons can exist and how many synapses can form.

/// A metabolic budget for one region.
#[derive(Debug, Clone)]
pub struct MetabolicBudget {
    /// Total neuron count allowed for this region.
    pub neuron_budget: u32,
    /// Maximum synapses per neuron (fanout cap).
    pub synapse_cap: u16,
    /// Volume dimensions [width, height, depth] in world units.
    pub volume: [f32; 3],
}

impl MetabolicBudget {
    /// Compute a metabolic budget from a proportion symbol and genome factors.
    ///
    /// - `proportion`: The symbolic proportion (`:large`, `:medium`, `:small`, `:tiny`)
    /// - `genome_dev_factor`: Genome DEV prior modulation (0.5 = half, 1.0 = neutral, 2.0 = double)
    /// - `base_neuron_pool`: The organism-wide base neuron count per region
    pub fn from_proportion(
        proportion: Option<&str>,
        genome_dev_factor: f32,
        base_neuron_pool: u32,
    ) -> Self {
        let (fraction, volume) = match proportion {
            Some("large") => (1.0, [10.0, 6.0, 3.0]),
            Some("medium") => (0.6, [8.0, 5.0, 2.5]),
            Some("small") => (0.3, [5.0, 3.0, 2.0]),
            Some("tiny") => (0.15, [3.0, 2.0, 1.5]),
            _ => (0.6, [8.0, 5.0, 2.5]), // default to medium
        };

        let raw_neurons = (base_neuron_pool as f32 * fraction * genome_dev_factor) as u32;
        let neuron_budget = raw_neurons.max(8); // minimum 8 neurons

        MetabolicBudget {
            neuron_budget,
            synapse_cap: 20,
            volume,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_gets_more_than_small() {
        let large = MetabolicBudget::from_proportion(Some("large"), 1.0, 1000);
        let small = MetabolicBudget::from_proportion(Some("small"), 1.0, 1000);

        assert!(
            large.neuron_budget > small.neuron_budget,
            "large ({}) should exceed small ({})",
            large.neuron_budget,
            small.neuron_budget
        );
    }

    #[test]
    fn genome_modulates_budget() {
        let normal = MetabolicBudget::from_proportion(Some("medium"), 1.0, 1000);
        let boosted = MetabolicBudget::from_proportion(Some("medium"), 1.5, 1000);

        assert!(
            boosted.neuron_budget > normal.neuron_budget,
            "boosted ({}) should exceed normal ({})",
            boosted.neuron_budget,
            normal.neuron_budget
        );
    }

    #[test]
    fn minimum_enforced() {
        let tiny = MetabolicBudget::from_proportion(Some("tiny"), 0.01, 10);
        assert!(tiny.neuron_budget >= 8, "minimum should be 8, got {}", tiny.neuron_budget);
    }

    #[test]
    fn default_is_medium() {
        let default = MetabolicBudget::from_proportion(None, 1.0, 1000);
        let medium = MetabolicBudget::from_proportion(Some("medium"), 1.0, 1000);
        assert_eq!(default.neuron_budget, medium.neuron_budget);
    }
}
