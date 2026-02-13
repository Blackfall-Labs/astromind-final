//! Imaginal disc differentiation engine.
//!
//! Ported from pool-ternsig-experiment's disc.rs with key v2 change:
//! disc anchors are derived from trophic field sampling, not hardcoded positions.
//!
//! The commitment ratchet: pressure only accumulates, never decreases.
//! Once a neuron crosses threshold for a disc, it commits irreversibly.

use neuropool::spatial::{Nuclei, SpatialNeuron};

use crate::trophic::field::TrophicField;

/// A disc program — defines a differentiation attractor in the tissue.
///
/// Neurons near the anchor accumulate pressure toward this disc.
/// When pressure crosses threshold, the neuron commits and its nuclei
/// blend toward the disc's target nuclei.
#[derive(Debug, Clone)]
pub struct DiscProgram {
    /// Human-readable name (e.g., "sensory_input", "motor_output").
    pub name: String,
    /// Target nuclei — what committed neurons converge toward.
    pub target_nuclei: Nuclei,
    /// Spatial anchor — center of influence in world coordinates.
    /// Derived from trophic field maxima, NOT hardcoded.
    pub anchor: Option<[f32; 3]>,
    /// Influence radius in world units.
    pub radius: f32,
    /// Pressure threshold to trigger commitment.
    pub threshold: f32,
    /// Maximum fraction of total neurons this disc can claim (0.0-1.0).
    pub population_cap: f32,
    /// Whether committed neurons should be inhibitory.
    pub inhibitory: bool,
    /// Whether committed neurons should have recurrent wiring bias.
    pub recurrent: bool,
}

impl DiscProgram {
    /// Compute spatial influence at a position (0.0 outside radius, 1.0 at anchor).
    pub fn spatial_influence(&self, pos: [f32; 3]) -> f32 {
        let anchor = match self.anchor {
            Some(a) => a,
            None => return 0.0, // null anchor = disc never fires
        };

        let dx = pos[0] - anchor[0];
        let dy = pos[1] - anchor[1];
        let dz = pos[2] - anchor[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist >= self.radius {
            0.0
        } else {
            1.0 - (dist / self.radius)
        }
    }

    /// Compute trophic influence at a position from a trophic field.
    pub fn trophic_influence(&self, pos: [f32; 3], field: Option<&TrophicField>) -> f32 {
        match field {
            Some(f) => f.sample(pos) as f32 / 65535.0,
            None => 0.0,
        }
    }
}

/// Tracks differentiation pressure for each neuron × disc pair.
pub struct DifferentiationState {
    neuron_count: usize,
    disc_count: usize,
    /// Pressure matrix [neuron_idx * disc_count + disc_idx].
    pressure: Vec<f32>,
    /// Whether each neuron has committed.
    pub differentiated: Vec<bool>,
    /// Which disc each neuron committed to (None if uncommitted).
    pub committed_disc: Vec<Option<usize>>,
}

impl DifferentiationState {
    pub fn new(neuron_count: usize, disc_count: usize) -> Self {
        DifferentiationState {
            neuron_count,
            disc_count,
            pressure: vec![0.0; neuron_count * disc_count],
            differentiated: vec![false; neuron_count],
            committed_disc: vec![None; neuron_count],
        }
    }

    /// Get pressure for a specific neuron and disc.
    pub fn pressure(&self, neuron_idx: usize, disc_idx: usize) -> f32 {
        self.pressure[neuron_idx * self.disc_count + disc_idx]
    }

    /// Accumulate pressure (ratchet — only increases).
    pub fn accumulate(&mut self, neuron_idx: usize, disc_idx: usize, amount: f32) {
        if amount > 0.0 {
            self.pressure[neuron_idx * self.disc_count + disc_idx] += amount;
        }
    }

    /// Find the winning disc for a neuron (highest pressure above threshold).
    pub fn winning_disc(&self, neuron_idx: usize, discs: &[DiscProgram]) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (di, disc) in discs.iter().enumerate() {
            let p = self.pressure(neuron_idx, di);
            if p >= disc.threshold {
                match best {
                    None => best = Some((di, p)),
                    Some((_, bp)) if p > bp => best = Some((di, p)),
                    _ => {}
                }
            }
        }
        best.map(|(idx, _)| idx)
    }

    pub fn differentiated_count(&self) -> usize {
        self.differentiated.iter().filter(|&&d| d).count()
    }

    /// Count how many neurons committed to each disc.
    pub fn disc_populations(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.disc_count];
        for &disc in self.committed_disc.iter().flatten() {
            if disc < self.disc_count {
                counts[disc] += 1;
            }
        }
        counts
    }
}

/// Accumulate spatial + trophic pressure for one round.
///
/// For each undifferentiated neuron, for each disc:
/// - Compute spatial influence (proximity to anchor)
/// - Compute trophic influence (chemical concentration at position)
/// - Combined pressure = spatial * 0.6 + trophic * 0.4
/// - Accumulate (ratchet — never decreases)
pub fn accumulate_pressure(
    neurons: &[SpatialNeuron],
    discs: &[DiscProgram],
    trophic_fields: &[Option<&TrophicField>],
    state: &mut DifferentiationState,
    rate: f32,
) {
    for (ni, neuron) in neurons.iter().enumerate() {
        if state.differentiated[ni] {
            continue;
        }

        let pos = neuron.soma.position;

        for (di, disc) in discs.iter().enumerate() {
            let spatial = disc.spatial_influence(pos);
            let trophic = if di < trophic_fields.len() {
                disc.trophic_influence(pos, trophic_fields[di])
            } else {
                0.0
            };

            let combined = (spatial * 0.6 + trophic * 0.4) * rate;
            state.accumulate(ni, di, combined);
        }
    }
}

/// Differentiate neurons that have crossed threshold.
///
/// Returns vec of (neuron_idx, disc_idx) pairs for newly committed neurons.
/// Respects population caps.
pub fn differentiate(
    neuron_count: usize,
    discs: &[DiscProgram],
    state: &mut DifferentiationState,
    max_per_round: usize,
) -> Vec<(usize, usize)> {
    let mut committed_this_round = Vec::new();
    let current_pops = state.disc_populations();

    for ni in 0..neuron_count {
        if state.differentiated[ni] {
            continue;
        }
        if committed_this_round.len() >= max_per_round {
            break;
        }

        if let Some(di) = state.winning_disc(ni, discs) {
            // Check population cap
            let cap_count =
                (discs[di].population_cap * neuron_count as f32).ceil() as usize;
            if current_pops[di] + committed_this_round.iter().filter(|(_, d)| *d == di).count()
                < cap_count
            {
                state.differentiated[ni] = true;
                state.committed_disc[ni] = Some(di);
                committed_this_round.push((ni, di));
            }
        }
    }

    committed_this_round
}

/// Blend a neuron's nuclei toward a disc's target nuclei.
///
/// `blend` controls convergence rate: 0.5 = move halfway toward target each call.
pub fn blend_nuclei(neuron: &mut SpatialNeuron, disc: &DiscProgram, blend: f32) {
    let target = &disc.target_nuclei;
    let n = &mut neuron.nuclei;

    n.soma_size = lerp_u8(n.soma_size, target.soma_size, blend);
    n.axon_affinity = lerp_u8(n.axon_affinity, target.axon_affinity, blend);
    n.myelin_affinity = lerp_u8(n.myelin_affinity, target.myelin_affinity, blend);
    n.metabolic_rate = lerp_u8(n.metabolic_rate, target.metabolic_rate, blend);
    n.leak = lerp_u8(n.leak, target.leak, blend);
    n.refractory = lerp_u32(n.refractory, target.refractory, blend);

    // Interface and polarity snap — no blending
    n.interface = target.interface;
    if disc.inhibitory {
        n.polarity = neuropool::Polarity::Negative;
    } else {
        n.polarity = target.polarity;
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let result = a as f32 * (1.0 - t) + b as f32 * t;
    result.round().clamp(0.0, 255.0) as u8
}

fn lerp_u32(a: u32, b: u32, t: f32) -> u32 {
    let result = a as f32 * (1.0 - t) + b as f32 * t;
    result.round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuropool::spatial::{Interface, Nuclei, Polarity};

    fn make_disc(name: &str, anchor: [f32; 3], radius: f32) -> DiscProgram {
        DiscProgram {
            name: name.to_string(),
            target_nuclei: Nuclei::sensory(0, 0),
            anchor: Some(anchor),
            radius,
            threshold: 5.0,
            population_cap: 0.3,
            inhibitory: false,
            recurrent: false,
        }
    }

    fn make_neuron(pos: [f32; 3]) -> SpatialNeuron {
        SpatialNeuron::pyramidal_at(pos)
    }

    #[test]
    fn spatial_influence_at_anchor_is_one() {
        let disc = make_disc("test", [5.0, 5.0, 5.0], 3.0);
        let inf = disc.spatial_influence([5.0, 5.0, 5.0]);
        assert!((inf - 1.0).abs() < 0.001);
    }

    #[test]
    fn spatial_influence_outside_radius_is_zero() {
        let disc = make_disc("test", [5.0, 5.0, 5.0], 3.0);
        let inf = disc.spatial_influence([20.0, 20.0, 20.0]);
        assert!(inf == 0.0);
    }

    #[test]
    fn null_anchor_never_fires() {
        let mut disc = make_disc("test", [0.0, 0.0, 0.0], 5.0);
        disc.anchor = None;
        let inf = disc.spatial_influence([0.0, 0.0, 0.0]);
        assert_eq!(inf, 0.0);
    }

    #[test]
    fn pressure_accumulates() {
        let neurons = vec![make_neuron([5.0, 5.0, 5.0])];
        let discs = vec![make_disc("test", [5.0, 5.0, 5.0], 3.0)];
        let mut state = DifferentiationState::new(1, 1);

        // Run several rounds
        for _ in 0..10 {
            accumulate_pressure(&neurons, &discs, &[], &mut state, 1.0);
        }

        let p = state.pressure(0, 0);
        assert!(p > 0.0, "pressure should accumulate: {p}");
    }

    #[test]
    fn differentiation_commits_neurons() {
        let neurons: Vec<_> = (0..10)
            .map(|i| make_neuron([5.0 + i as f32 * 0.1, 5.0, 5.0]))
            .collect();
        let discs = vec![make_disc("test", [5.0, 5.0, 5.0], 5.0)];
        let mut state = DifferentiationState::new(10, 1);

        // Accumulate enough pressure
        for _ in 0..20 {
            accumulate_pressure(&neurons, &discs, &[], &mut state, 1.0);
        }

        let committed = differentiate(10, &discs, &mut state, 100);
        assert!(
            !committed.is_empty(),
            "some neurons should commit after sufficient pressure"
        );
        assert!(
            state.differentiated_count() > 0,
            "differentiated count should increase"
        );
    }

    #[test]
    fn population_cap_respected() {
        let neurons: Vec<_> = (0..100)
            .map(|i| make_neuron([5.0 + (i % 10) as f32 * 0.1, 5.0, 5.0]))
            .collect();
        let mut disc = make_disc("test", [5.0, 5.0, 5.0], 10.0);
        disc.population_cap = 0.1; // max 10 neurons
        disc.threshold = 0.1; // low threshold

        let mut state = DifferentiationState::new(100, 1);

        // Accumulate lots of pressure
        for _ in 0..50 {
            accumulate_pressure(&neurons, &[disc.clone()], &[], &mut state, 1.0);
        }

        // Try to differentiate all
        differentiate(100, &[disc.clone()], &mut state, 100);

        let pops = state.disc_populations();
        assert!(
            pops[0] <= 10,
            "population cap of 10% should limit to 10, got {}",
            pops[0]
        );
    }

    #[test]
    fn blend_moves_toward_target() {
        let mut neuron = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        let original_soma = neuron.nuclei.soma_size;

        let disc = DiscProgram {
            name: "sensory".to_string(),
            target_nuclei: Nuclei::new(
                50, 50, 50, 50, 50, 2000, 0,
                Interface::sensory(0, 0),
                Polarity::Positive,
            ),
            anchor: Some([0.0, 0.0, 0.0]),
            radius: 5.0,
            threshold: 1.0,
            population_cap: 0.5,
            inhibitory: false,
            recurrent: false,
        };

        blend_nuclei(&mut neuron, &disc, 0.5);

        // Should have moved toward 50
        let diff_before = (original_soma as i32 - 50).unsigned_abs();
        let diff_after = (neuron.nuclei.soma_size as i32 - 50).unsigned_abs();
        assert!(
            diff_after < diff_before,
            "soma should move toward target: was {original_soma}, now {}, target 50",
            neuron.nuclei.soma_size
        );
    }
}
