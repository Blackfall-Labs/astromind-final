//! Intensity symbol resolution.
//!
//! Maps symbolic intensity levels (`:dominant`, `:moderate`, `:sparse`, etc.)
//! to priority levels. Higher priority claims more of the metabolic budget.

use crate::parser::ast::{PopulateBlock, Symbol};

/// Priority level — lower number = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntensityLevel(pub u8);

impl IntensityLevel {
    /// Map a symbol to a priority level.
    /// Returns None for unrecognized symbols.
    pub fn from_symbol(sym: &Symbol) -> Option<Self> {
        match sym.as_str() {
            "dominant" => Some(IntensityLevel(1)),
            "major" => Some(IntensityLevel(2)),
            "moderate" => Some(IntensityLevel(3)),
            "minor" => Some(IntensityLevel(4)),
            "sparse" => Some(IntensityLevel(5)),
            "very_rare" | "rare" => Some(IntensityLevel(6)),
            _ => None,
        }
    }

    /// Base fraction of total budget this priority claims before competition.
    pub fn base_fraction(self) -> f32 {
        match self.0 {
            1 => 0.40, // dominant
            2 => 0.25, // major
            3 => 0.15, // moderate
            4 => 0.10, // minor
            5 => 0.06, // sparse
            6 => 0.04, // very_rare
            _ => 0.02,
        }
    }
}

/// A resolved population entry — cell type with concrete neuron count.
#[derive(Debug, Clone)]
pub struct ResolvedPopulationEntry {
    pub cell_type: String,
    pub priority: IntensityLevel,
    pub neuron_count: u32,
}

/// Fully resolved population mix for a region.
#[derive(Debug, Clone)]
pub struct ResolvedPopulation {
    pub entries: Vec<ResolvedPopulationEntry>,
    pub total_neurons: u32,
}

/// Resolve a populate block into concrete neuron counts given a total budget.
///
/// Algorithm:
/// 1. Map each entry's intensity symbol to a priority level
/// 2. Compute base fractions, normalize to sum to 1.0
/// 3. Multiply by total budget, round to nearest integer
/// 4. Adjust remainder to match total exactly (give to highest priority)
pub fn resolve_population(populate: &PopulateBlock, total_budget: u32) -> ResolvedPopulation {
    let mut entries: Vec<(String, IntensityLevel)> = Vec::new();

    for entry in &populate.entries {
        let level = IntensityLevel::from_symbol(&entry.intensity).unwrap_or(IntensityLevel(3));
        entries.push((entry.cell_type.as_str().to_string(), level));
    }

    if entries.is_empty() {
        return ResolvedPopulation {
            entries: Vec::new(),
            total_neurons: 0,
        };
    }

    // Compute base fractions and normalize
    let raw_total: f32 = entries.iter().map(|(_, lvl)| lvl.base_fraction()).sum();
    let scale = if raw_total > 0.0 {
        1.0 / raw_total
    } else {
        1.0
    };

    let mut resolved: Vec<ResolvedPopulationEntry> = Vec::new();
    let mut allocated = 0u32;

    for (cell_type, priority) in &entries {
        let fraction = priority.base_fraction() * scale;
        let count = (fraction * total_budget as f32).round() as u32;
        allocated += count;
        resolved.push(ResolvedPopulationEntry {
            cell_type: cell_type.clone(),
            priority: *priority,
            neuron_count: count,
        });
    }

    // Fix rounding: give remainder to highest priority entry
    if allocated != total_budget && !resolved.is_empty() {
        // Find highest priority (lowest number)
        let best_idx = resolved
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.priority)
            .map(|(i, _)| i)
            .unwrap_or(0);

        if allocated < total_budget {
            resolved[best_idx].neuron_count += total_budget - allocated;
        } else if resolved[best_idx].neuron_count >= (allocated - total_budget) {
            resolved[best_idx].neuron_count -= allocated - total_budget;
        }
    }

    let total = resolved.iter().map(|e| e.neuron_count).sum();

    ResolvedPopulation {
        entries: resolved,
        total_neurons: total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{PopulateEntry, Span};

    fn entry(cell_type: &str, intensity: &str) -> PopulateEntry {
        PopulateEntry {
            cell_type: Symbol(cell_type.to_string()),
            intensity: Symbol(intensity.to_string()),
            span: Span::default(),
        }
    }

    #[test]
    fn basic_resolution() {
        let block = PopulateBlock {
            entries: vec![
                entry("pyramidal", "dominant"),
                entry("inhibitory", "moderate"),
                entry("gate", "sparse"),
            ],
            span: Span::default(),
        };

        let resolved = resolve_population(&block, 1000);

        assert_eq!(resolved.total_neurons, 1000);
        assert!(resolved.entries.len() == 3);

        // Dominant should have most neurons
        let pyramidal = &resolved.entries[0];
        let inhibitory = &resolved.entries[1];
        let gate = &resolved.entries[2];

        assert!(
            pyramidal.neuron_count > inhibitory.neuron_count,
            "pyramidal ({}) should exceed inhibitory ({})",
            pyramidal.neuron_count,
            inhibitory.neuron_count
        );
        assert!(
            inhibitory.neuron_count > gate.neuron_count,
            "inhibitory ({}) should exceed gate ({})",
            inhibitory.neuron_count,
            gate.neuron_count
        );
    }

    #[test]
    fn empty_populate_gives_zero() {
        let block = PopulateBlock {
            entries: Vec::new(),
            span: Span::default(),
        };
        let resolved = resolve_population(&block, 500);
        assert_eq!(resolved.total_neurons, 0);
    }

    #[test]
    fn single_entry_gets_all() {
        let block = PopulateBlock {
            entries: vec![entry("pyramidal", "moderate")],
            span: Span::default(),
        };
        let resolved = resolve_population(&block, 200);
        assert_eq!(resolved.total_neurons, 200);
        assert_eq!(resolved.entries[0].neuron_count, 200);
    }

    #[test]
    fn priority_ordering() {
        assert!(IntensityLevel(1) < IntensityLevel(2));
        assert!(IntensityLevel::from_symbol(&Symbol("dominant".to_string())).unwrap().0 == 1);
        assert!(IntensityLevel::from_symbol(&Symbol("sparse".to_string())).unwrap().0 == 5);
    }
}
