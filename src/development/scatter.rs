//! Neuron scattering — places neurons in 3D space with type-appropriate nuclei.
//!
//! Ported from pool-ternsig-experiment's scatter_m1_neurons, generalized for
//! any region shape and population mix from a resolved populate block.

use neuropool::spatial::{Axon, Dendrite, Interface, Nuclei, Polarity, Soma, SpatialNeuron};

use crate::resolve::genome::GenomeProfile;
use crate::resolve::intensity::ResolvedPopulation;

/// Scatter neurons into a volume according to a resolved population mix.
///
/// - `population`: Resolved population with concrete neuron counts per type.
/// - `volume`: [width, height, depth] of the region volume in world units.
/// - `genome`: Genome profile for modulating nuclei properties.
/// - `seed`: Deterministic seed for reproducible placement.
///
/// Returns a Vec of spatial neurons placed within the volume.
pub fn scatter_neurons(
    population: &ResolvedPopulation,
    volume: [f32; 3],
    genome: &GenomeProfile,
    seed: u64,
) -> Vec<SpatialNeuron> {
    let mut neurons = Vec::with_capacity(population.total_neurons as usize);
    let mut rng = SimpleRng::new(seed);

    for entry in &population.entries {
        let nuclei_template = nuclei_for_type(&entry.cell_type, genome);

        for _ in 0..entry.neuron_count {
            let pos = [
                rng.next_f32() * volume[0],
                rng.next_f32() * volume[1],
                rng.next_f32() * volume[2],
            ];

            // Jitter nuclei properties +-15%
            let nuclei = jitter_nuclei(&nuclei_template, &mut rng);

            // Axon points roughly toward volume center with some randomness
            let center = [volume[0] * 0.5, volume[1] * 0.5, volume[2] * 0.5];
            let ax_target = [
                pos[0] + (center[0] - pos[0]) * 0.3 + (rng.next_f32() - 0.5) * volume[0] * 0.4,
                pos[1] + (center[1] - pos[1]) * 0.3 + (rng.next_f32() - 0.5) * volume[1] * 0.4,
                pos[2] + (center[2] - pos[2]) * 0.3 + (rng.next_f32() - 0.5) * volume[2] * 0.4,
            ];

            let dendrite_radius = 1.0 + (nuclei.soma_size as f32 / 255.0) * 2.0;
            let spine_count = nuclei.max_dendrite_connections();

            let neuron = SpatialNeuron::new(
                Soma::at(pos),
                Dendrite::new(dendrite_radius, spine_count),
                Axon::toward(ax_target),
                nuclei,
            );
            neurons.push(neuron);
        }
    }

    neurons
}

/// Map a cell type name to base nuclei properties.
fn nuclei_for_type(cell_type: &str, genome: &GenomeProfile) -> Nuclei {
    match cell_type {
        "pyramidal" | "excitatory" => Nuclei::pyramidal(),

        "inhibitory" => Nuclei::interneuron(),

        "relay" => Nuclei::relay(),

        "gate" => {
            let mut n = Nuclei::gate();
            // Governance modulates gate leak (stronger gates with high GOV)
            let leak_mod = (n.leak as f32 * genome.gate_fraction_modifier()).min(255.0) as u8;
            n.leak = leak_mod;
            n
        }

        "sensory" => Nuclei::sensory(0, 0), // channel/modality set during differentiation

        "motor" => Nuclei::motor(0, 0), // channel/modality set during differentiation

        "modulator" => Nuclei::modulator(0),

        "oscillator" => Nuclei::oscillator(50_000), // 50ms default period

        "granule" => {
            // Granule cells: small, sparse, high threshold — like hippocampal DG
            Nuclei::new(
                60,  // small soma
                30,  // short axon
                40,  // low myelin
                80,  // moderate metabolic
                150, // high leak (fast decay)
                3000,
                0,
                Interface::none(),
                Polarity::Positive,
            )
        }

        "memory_reader" => Nuclei::memory(0),

        "memory_writer" => Nuclei::memory(0),

        // Default to interneuron-like properties
        _ => Nuclei::interneuron(),
    }
}

/// Apply +-15% jitter to nuclei properties for neuronal diversity.
fn jitter_nuclei(base: &Nuclei, rng: &mut SimpleRng) -> Nuclei {
    Nuclei::new(
        jitter_u8(base.soma_size, rng),
        jitter_u8(base.axon_affinity, rng),
        jitter_u8(base.myelin_affinity, rng),
        jitter_u8(base.metabolic_rate, rng),
        jitter_u8(base.leak, rng),
        jitter_u32(base.refractory, rng),
        base.oscillation_period, // oscillation period is exact, not jittered
        base.interface,
        base.polarity,
    )
}

fn jitter_u8(val: u8, rng: &mut SimpleRng) -> u8 {
    let factor = 0.85 + rng.next_f32() * 0.30; // 0.85 to 1.15
    (val as f32 * factor).round().clamp(0.0, 255.0) as u8
}

fn jitter_u32(val: u32, rng: &mut SimpleRng) -> u32 {
    let factor = 0.85 + rng.next_f32() * 0.30;
    (val as f32 * factor).round() as u32
}

/// Minimal deterministic PRNG (xorshift64) — no external rand dependency.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{PopulateBlock, PopulateEntry, Span, Symbol};
    use crate::resolve::intensity::resolve_population;

    fn make_population(entries: &[(&str, &str)], total: u32) -> ResolvedPopulation {
        let block = PopulateBlock {
            entries: entries
                .iter()
                .map(|(ct, int)| PopulateEntry {
                    cell_type: Symbol(ct.to_string()),
                    intensity: Symbol(int.to_string()),
                    span: Span::default(),
                })
                .collect(),
            span: Span::default(),
        };
        resolve_population(&block, total)
    }

    #[test]
    fn scatter_produces_correct_count() {
        let pop = make_population(
            &[
                ("pyramidal", "dominant"),
                ("inhibitory", "moderate"),
                ("gate", "sparse"),
            ],
            500,
        );

        let neurons = scatter_neurons(&pop, [10.0, 6.0, 3.0], &GenomeProfile::neutral(), 42);
        assert_eq!(neurons.len(), 500, "should scatter exactly 500 neurons");
    }

    #[test]
    fn neurons_within_volume() {
        let pop = make_population(&[("pyramidal", "dominant")], 100);
        let volume = [10.0, 6.0, 3.0];
        let neurons = scatter_neurons(&pop, volume, &GenomeProfile::neutral(), 123);

        for n in &neurons {
            let p = n.soma.position;
            assert!(p[0] >= 0.0 && p[0] <= volume[0], "x out of bounds: {}", p[0]);
            assert!(p[1] >= 0.0 && p[1] <= volume[1], "y out of bounds: {}", p[1]);
            assert!(p[2] >= 0.0 && p[2] <= volume[2], "z out of bounds: {}", p[2]);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let pop = make_population(&[("pyramidal", "dominant")], 50);
        let a = scatter_neurons(&pop, [5.0, 5.0, 5.0], &GenomeProfile::neutral(), 42);
        let b = scatter_neurons(&pop, [5.0, 5.0, 5.0], &GenomeProfile::neutral(), 42);

        for (na, nb) in a.iter().zip(b.iter()) {
            assert_eq!(na.soma.position, nb.soma.position);
        }
    }

    #[test]
    fn different_seeds_different_placement() {
        let pop = make_population(&[("pyramidal", "dominant")], 50);
        let a = scatter_neurons(&pop, [5.0, 5.0, 5.0], &GenomeProfile::neutral(), 42);
        let b = scatter_neurons(&pop, [5.0, 5.0, 5.0], &GenomeProfile::neutral(), 99);

        let different = a
            .iter()
            .zip(b.iter())
            .any(|(na, nb)| na.soma.position != nb.soma.position);
        assert!(different, "different seeds should produce different placement");
    }

    #[test]
    fn nuclei_diversity_via_jitter() {
        let pop = make_population(&[("pyramidal", "dominant")], 100);
        let neurons = scatter_neurons(&pop, [10.0, 6.0, 3.0], &GenomeProfile::neutral(), 42);

        // Not all soma sizes should be identical due to jitter
        let soma_sizes: Vec<u8> = neurons.iter().map(|n| n.nuclei.soma_size).collect();
        let unique: std::collections::HashSet<u8> = soma_sizes.iter().copied().collect();
        assert!(
            unique.len() > 1,
            "jitter should produce diverse soma sizes, got {} unique",
            unique.len()
        );
    }
}
