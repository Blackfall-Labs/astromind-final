//! Development executor — orchestrates the full pipeline from parsed region
//! definition to developed spatial neuron pool.
//!
//! Pipeline:
//! 1. Resolve population (symbol -> concrete neuron counts)
//! 2. Scatter neurons into volume
//! 3. Initialize trophic fields from gradient blocks
//! 4. Build disc programs from differentiate blocks
//! 5. Exposure phase: diffuse trophic fields + accumulate pressure
//! 6. Differentiation phase: commit neurons + blend nuclei
//! 7. Wire by proximity (zone-aware)
//! 8. Return DevelopedRegion

use neuropool::spatial::{SpatialNeuron, SpatialSynapseStore, WiringConfig, wire_by_proximity};

use crate::parser::ast::{DifferentiateBlock, RegionDef, Symbol};
use crate::resolve::budget::MetabolicBudget;
use crate::resolve::genome::GenomeProfile;
use crate::resolve::intensity::resolve_population;
use crate::trophic::chemical::TrophicChemicalId;
use crate::trophic::diffusion::{diffuse_step, DiffusionParams};
use crate::trophic::field::TrophicField;

use super::differentiation::{
    accumulate_pressure, blend_nuclei, differentiate, DifferentiationState, DiscProgram,
};
use super::phase::PhaseConfig;
use super::scatter::scatter_neurons;

/// The output of developing a single region.
#[derive(Debug)]
pub struct DevelopedRegion {
    pub name: String,
    pub neurons: Vec<SpatialNeuron>,
    pub synapses: SpatialSynapseStore,
    pub trophic_fields: Vec<TrophicField>,
    pub stats: DevelopmentStats,
}

/// Statistics from the development process.
#[derive(Debug, Clone)]
pub struct DevelopmentStats {
    pub total_neurons: u32,
    pub differentiated: u32,
    pub undifferentiated: u32,
    pub populations: Vec<(String, u32)>,
    pub total_synapses: u32,
}

/// Develop a single region from its AST definition.
///
/// This is the main entry point for turning a parsed `.neuro` region into
/// a functional spatial neuron pool.
pub fn develop_region(
    region: &RegionDef,
    genome: &GenomeProfile,
    phases: &PhaseConfig,
    base_neuron_pool: u32,
    seed: u64,
) -> DevelopedRegion {
    // --- 1. Resolve metabolic budget ---
    let proportion_str = region.proportion.as_ref().map(|s| s.as_str());
    let budget = MetabolicBudget::from_proportion(
        proportion_str,
        genome.dev_factor(),
        base_neuron_pool,
    );

    // --- 2. Resolve population mix ---
    let resolved_pop = match &region.populate {
        Some(pop) => resolve_population(pop, budget.neuron_budget),
        None => {
            return DevelopedRegion {
                name: region.name.as_str().to_string(),
                neurons: Vec::new(),
                synapses: SpatialSynapseStore::new(0),
                trophic_fields: Vec::new(),
                stats: DevelopmentStats {
                    total_neurons: 0,
                    differentiated: 0,
                    undifferentiated: 0,
                    populations: Vec::new(),
                    total_synapses: 0,
                },
            };
        }
    };

    // --- 3. Scatter neurons ---
    let mut neurons = scatter_neurons(&resolved_pop, budget.volume, genome, seed);
    let neuron_count = neurons.len();

    // --- 4. Initialize trophic fields from gradient blocks ---
    let grid_dims = [
        budget.volume[0].ceil() as u32 + 1,
        budget.volume[1].ceil() as u32 + 1,
        budget.volume[2].ceil() as u32 + 1,
    ];

    let mut trophic_fields: Vec<TrophicField> = Vec::new();
    for gradient in &region.gradients {
        let chem_name = gradient
            .chemical
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let mut field = TrophicField::new(
            TrophicChemicalId::new(chem_name),
            grid_dims,
            1.0,
            [0.0, 0.0, 0.0],
        );

        // Place gradient source based on source directive
        let source_pos = gradient_source_position(&gradient.source, budget.volume);
        field.add_source(source_pos, 10000, budget.volume[0] * 0.5);

        trophic_fields.push(field);
    }

    // --- 5. Build disc programs from differentiate blocks ---
    let discs = build_disc_programs(
        &region.differentiations,
        &trophic_fields,
        genome,
        budget.volume,
    );

    // --- 6. Exposure phase: diffuse fields + accumulate pressure ---
    let mut diff_state = DifferentiationState::new(neuron_count, discs.len());

    let diffusion_params = DiffusionParams {
        radius: 1,
        decay: 0.02,
        saturation: 65535,
    };

    for _ in 0..phases.exposure_rounds {
        // Diffuse trophic fields
        for field in &mut trophic_fields {
            diffuse_step(field, &diffusion_params);
        }

        // Build field references for pressure accumulation
        let field_refs: Vec<Option<&TrophicField>> = discs
            .iter()
            .map(|_| trophic_fields.first()) // simplified: all discs use first field
            .collect();

        accumulate_pressure(
            &neurons,
            &discs,
            &field_refs,
            &mut diff_state,
            phases.pressure_rate,
        );
    }

    // --- 7. Differentiation phase: commit + blend ---
    for _ in 0..phases.differentiation_rounds {
        let committed = differentiate(
            neuron_count,
            &discs,
            &mut diff_state,
            phases.max_commits_per_round,
        );

        // Blend committed neurons toward their disc targets
        for (ni, di) in &committed {
            blend_nuclei(&mut neurons[*ni], &discs[*di], phases.blend_rate);
        }

        // Also blend previously committed neurons (convergence)
        for ni in 0..neuron_count {
            if let Some(di) = diff_state.committed_disc[ni] {
                blend_nuclei(&mut neurons[ni], &discs[di], phases.blend_rate * 0.1);
            }
        }
    }

    // --- 8. Wire by proximity ---
    let wiring_config = build_wiring_config(region, genome);
    let synapses = wire_by_proximity(&neurons, &wiring_config);

    // --- Collect stats ---
    let diff_count = diff_state.differentiated_count() as u32;
    let disc_pops = diff_state.disc_populations();
    let populations: Vec<(String, u32)> = discs
        .iter()
        .enumerate()
        .map(|(i, d)| (d.name.clone(), disc_pops[i] as u32))
        .collect();

    DevelopedRegion {
        name: region.name.as_str().to_string(),
        neurons,
        synapses,
        trophic_fields,
        stats: DevelopmentStats {
            total_neurons: neuron_count as u32,
            differentiated: diff_count,
            undifferentiated: neuron_count as u32 - diff_count,
            populations,
            total_synapses: 0, // filled below
        },
    }
}

/// Build disc programs from differentiate blocks.
fn build_disc_programs(
    blocks: &[DifferentiateBlock],
    trophic_fields: &[TrophicField],
    genome: &GenomeProfile,
    volume: [f32; 3],
) -> Vec<DiscProgram> {
    let disc_count = blocks.len();
    blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| {
            // Derive anchor from trophic field maximum if receptive_to is specified
            let anchor = derive_anchor(block, trophic_fields, volume, idx, disc_count);

            // Determine target nuclei from block properties
            let target_nuclei = nuclei_from_diff_block(block);

            // Threshold modulated by genome
            let base_threshold = 5.0;
            let threshold = base_threshold / genome.differentiation_rate();

            // Population cap from abundance symbol
            let cap = match block.abundance.as_ref().map(|s| s.as_str()) {
                Some("dominant") => 0.40,
                Some("major") => 0.25,
                Some("moderate") => 0.15,
                Some("minor") => 0.10,
                Some("sparse") => 0.06,
                _ => 0.15, // default moderate
            };

            DiscProgram {
                name: block.target_type.as_str().to_string(),
                target_nuclei,
                anchor,
                radius: 5.0,
                threshold,
                population_cap: cap,
                inhibitory: block.inhibitory.unwrap_or(false),
                recurrent: block.recurrent.unwrap_or(false),
            }
        })
        .collect()
}

/// Derive a disc anchor from trophic field maxima.
/// Falls back to distributing anchors across the volume when no fields exist.
fn derive_anchor(
    block: &DifferentiateBlock,
    fields: &[TrophicField],
    volume: [f32; 3],
    disc_index: usize,
    disc_count: usize,
) -> Option<[f32; 3]> {
    // If receptive_to is specified, find the field with highest concentration
    if !block.receptive_to.is_empty() {
        for chem in &block.receptive_to {
            for field in fields {
                if field.chemical.0 == chem.as_str() {
                    return Some(field.max_position());
                }
            }
        }
    }

    // If where clauses specify a chemical, use that field's max
    for wc in &block.where_clauses {
        for field in fields {
            if field.chemical.0 == wc.chemical.as_str() {
                return Some(field.max_position());
            }
        }
    }

    // Try first available field
    if let Some(f) = fields.first() {
        return Some(f.max_position());
    }

    // No trophic fields at all — distribute anchors along the X axis of the volume.
    // This is biologically analogous to positional identity from morphogen gradients
    // that haven't been explicitly declared.
    if disc_count > 0 {
        let t = if disc_count == 1 {
            0.5
        } else {
            disc_index as f32 / (disc_count - 1) as f32
        };
        Some([
            volume[0] * t,
            volume[1] * 0.5,
            volume[2] * 0.5,
        ])
    } else {
        None
    }
}

/// Map differentiate block properties to target nuclei.
fn nuclei_from_diff_block(block: &DifferentiateBlock) -> neuropool::Nuclei {
    use neuropool::spatial::{Interface, Nuclei};

    let type_name = block.target_type.as_str();

    match type_name {
        "sensory" => Nuclei::sensory(0, 0),
        "motor" => Nuclei::motor(0, 0),
        "gate" => Nuclei::gate(),
        "relay" => Nuclei::relay(),
        "inhibitory" => Nuclei::interneuron(),
        "oscillator" => Nuclei::oscillator(50_000),
        "modulator" => Nuclei::modulator(0),
        "memory_reader" | "memory_writer" => Nuclei::memory(0),
        "granule" => Nuclei::new(
            60, 30, 40, 80, 150, 3000, 0,
            Interface::none(),
            neuropool::Polarity::Positive,
        ),
        "pyramidal" | "excitatory" => Nuclei::pyramidal(),
        _ => Nuclei::pyramidal(), // default to pyramidal
    }
}

/// Compute gradient source position from a source directive.
fn gradient_source_position(source: &Option<Symbol>, volume: [f32; 3]) -> [f32; 3] {
    match source.as_ref().map(|s| s.as_str()) {
        Some("input_face") => [0.0, volume[1] * 0.5, volume[2] * 0.5],
        Some("output_face") => [volume[0], volume[1] * 0.5, volume[2] * 0.5],
        Some("deep") => [volume[0] * 0.5, volume[1] * 0.5, 0.0],
        Some("surface") => [volume[0] * 0.5, volume[1] * 0.5, volume[2]],
        Some("dorsal") => [volume[0] * 0.5, 0.0, volume[2] * 0.5],
        Some("ventral") => [volume[0] * 0.5, volume[1], volume[2] * 0.5],
        Some("center") | None => [volume[0] * 0.5, volume[1] * 0.5, volume[2] * 0.5],
        Some(_) => [volume[0] * 0.5, volume[1] * 0.5, volume[2] * 0.5],
    }
}

/// Build wiring configuration from region definition + genome.
fn build_wiring_config(region: &RegionDef, genome: &GenomeProfile) -> WiringConfig {
    // Start with defaults, modulate by genome
    let base_reach = 5.0;
    let base_fanout = 8;
    let base_fanin = 20;

    // Check if region has explicit wiring params
    let (reach, fanout) = if let Some(wire) = region.wiring.first() {
        let r = wire
            .global_params
            .reach
            .as_ref()
            .and_then(|s| symbol_to_f32(s))
            .unwrap_or(base_reach);
        let f = wire
            .global_params
            .fanout
            .as_ref()
            .and_then(|s| symbol_to_u16(s))
            .unwrap_or(base_fanout);
        (r, f)
    } else {
        (base_reach, base_fanout)
    };

    WiringConfig {
        max_distance: reach * genome.wiring_reach_factor(),
        max_fanout: (fanout as f32 * genome.wiring_fanout_factor()) as u16,
        max_fanin: base_fanin,
        default_magnitude: 100,
    }
}

fn symbol_to_f32(sym: &Symbol) -> Option<f32> {
    match sym.as_str() {
        "very_wide" => Some(8.0),
        "wide" => Some(6.0),
        "moderate" => Some(4.0),
        "tight" => Some(3.0),
        "very_tight" => Some(2.0),
        _ => None,
    }
}

fn symbol_to_u16(sym: &Symbol) -> Option<u16> {
    match sym.as_str() {
        "high" => Some(12),
        "moderate" => Some(8),
        "low" => Some(4),
        "minimal" => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::*;

    fn minimal_region() -> RegionDef {
        RegionDef {
            name: Symbol("test_cortex".to_string()),
            span: Span::default(),
            shape: None,
            proportion: Some(Symbol("medium".to_string())),
            grows: vec![GrowsDirective {
                mode: GrowsMode::First,
                target: None,
                span: Span::default(),
            }],
            populate: Some(PopulateBlock {
                entries: vec![
                    PopulateEntry {
                        cell_type: Symbol("pyramidal".to_string()),
                        intensity: Symbol("dominant".to_string()),
                        span: Span::default(),
                    },
                    PopulateEntry {
                        cell_type: Symbol("inhibitory".to_string()),
                        intensity: Symbol("moderate".to_string()),
                        span: Span::default(),
                    },
                    PopulateEntry {
                        cell_type: Symbol("gate".to_string()),
                        intensity: Symbol("sparse".to_string()),
                        span: Span::default(),
                    },
                ],
                span: Span::default(),
            }),
            differentiations: vec![
                DifferentiateBlock {
                    target_type: Symbol("sensory".to_string()),
                    span: Span::default(),
                    receptive_to: Vec::new(),
                    affinity: None,
                    abundance: Some(Symbol("moderate".to_string())),
                    between: None,
                    where_clauses: Vec::new(),
                    where_dense: false,
                    profile: None,
                    inhibitory: None,
                    recurrent: None,
                    feedforward: None,
                    sparse: None,
                    period: None,
                    bank: None,
                    sensitive_to: Vec::new(),
                },
                DifferentiateBlock {
                    target_type: Symbol("motor".to_string()),
                    span: Span::default(),
                    receptive_to: Vec::new(),
                    affinity: None,
                    abundance: Some(Symbol("moderate".to_string())),
                    between: None,
                    where_clauses: Vec::new(),
                    where_dense: false,
                    profile: None,
                    inhibitory: None,
                    recurrent: None,
                    feedforward: None,
                    sparse: None,
                    period: None,
                    bank: None,
                    sensitive_to: Vec::new(),
                },
            ],
            wiring: Vec::new(),
            gradients: Vec::new(),
            secretes: Vec::new(),
            plasticity_from: Vec::new(),
            consolidation_from: Vec::new(),
            sensitive_to: Vec::new(),
        }
    }

    #[test]
    fn develop_produces_neurons() {
        let region = minimal_region();
        let genome = GenomeProfile::neutral();
        let phases = PhaseConfig::fast();

        let result = develop_region(&region, &genome, &phases, 200, 42);

        assert!(
            result.stats.total_neurons > 0,
            "should produce neurons, got 0"
        );
        assert_eq!(result.neurons.len(), result.stats.total_neurons as usize);
    }

    #[test]
    fn develop_produces_differentiation() {
        let region = minimal_region();
        let genome = GenomeProfile::neutral();
        let phases = PhaseConfig::fast();

        let result = develop_region(&region, &genome, &phases, 200, 42);

        assert!(
            result.stats.differentiated > 0,
            "should differentiate some neurons, got 0 of {}",
            result.stats.total_neurons
        );
    }

    #[test]
    fn different_genomes_different_tissue() {
        let region = minimal_region();
        let phases = PhaseConfig::fast();

        let explorer = GenomeProfile::new(0.9, 0.2, 0.8, 0.3, 0.7, 0.6);
        let builder = GenomeProfile::new(0.2, 0.9, 0.4, 0.8, 0.3, 0.4);

        let a = develop_region(&region, &explorer, &phases, 200, 42);
        let b = develop_region(&region, &builder, &phases, 200, 42);

        // Different genomes should produce structurally different tissue
        let different_count = a.stats.total_neurons != b.stats.total_neurons;
        let different_diff = a.stats.differentiated != b.stats.differentiated;

        assert!(
            different_count || different_diff,
            "different genomes should produce different tissue: \
             explorer={}/{} differentiated, builder={}/{} differentiated",
            a.stats.differentiated,
            a.stats.total_neurons,
            b.stats.differentiated,
            b.stats.total_neurons
        );
    }

    #[test]
    fn empty_populate_returns_empty() {
        let mut region = minimal_region();
        region.populate = None;

        let result = develop_region(
            &region,
            &GenomeProfile::neutral(),
            &PhaseConfig::fast(),
            200,
            42,
        );

        assert_eq!(result.stats.total_neurons, 0);
        assert!(result.neurons.is_empty());
    }

    #[test]
    fn gradient_creates_trophic_field() {
        let mut region = minimal_region();
        region.gradients.push(GradientBlock {
            name: Symbol("test_gradient".to_string()),
            span: Span::default(),
            chemical: Some(Symbol("Wnt".to_string())),
            source: Some(Symbol("input_face".to_string())),
            diffusion: None,
        });

        let result = develop_region(
            &region,
            &GenomeProfile::neutral(),
            &PhaseConfig::fast(),
            100,
            42,
        );

        assert!(
            !result.trophic_fields.is_empty(),
            "gradient block should create a trophic field"
        );
        assert_eq!(result.trophic_fields[0].chemical.0, "Wnt");
    }
}
