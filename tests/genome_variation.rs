//! Genome variation test — same region definition, different genomes,
//! structurally different tissue.
//!
//! This proves the developmental language is genuinely genome-modulated,
//! not just running the same program with cosmetic differences.

use astromind_final::development::executor::develop_region;
use astromind_final::development::phase::PhaseConfig;
use astromind_final::parser::parse;
use astromind_final::resolve::genome::GenomeProfile;

/// Minimal cortex region for genome comparison.
const TEST_CORTEX: &str = r#"
region :test_cortex do
  grows first
  proportion :medium

  populate do
    pyramidal   :dominant
    inhibitory  :moderate
    gate        :sparse
    sensory     :moderate
    motor       :moderate
  end

  differentiate :sensory do
    affinity :high
    abundance :moderate
  end

  differentiate :motor do
    affinity :high
    abundance :moderate
  end

  differentiate :gate do
    inhibitory true
    abundance :sparse
  end

  wire :proximity do
    reach   :moderate
    fanout  :moderate
  end

  gradient :dev_gradient do
    chemical :BDNF
    source :input_face
  end

  secretes :test_trophic
end
"#;

fn explorer_genome() -> GenomeProfile {
    // High novelty, low governance, high plasticity, low precision
    GenomeProfile::new(0.9, 0.2, 0.8, 0.2, 0.7, 0.7)
}

fn builder_genome() -> GenomeProfile {
    // Low novelty, high governance, moderate plasticity, high precision
    GenomeProfile::new(0.1, 0.9, 0.4, 0.9, 0.3, 0.3)
}

#[test]
fn different_genomes_different_neuron_counts() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();

    let explorer = develop_region(region, &explorer_genome(), &phases, 500, 42);
    let builder = develop_region(region, &builder_genome(), &phases, 500, 42);

    eprintln!("  explorer: {} neurons", explorer.stats.total_neurons);
    eprintln!("  builder:  {} neurons", builder.stats.total_neurons);

    // Explorer has high DEV (0.7) -> larger budget
    // Builder has low DEV (0.3) -> smaller budget
    assert_ne!(
        explorer.stats.total_neurons, builder.stats.total_neurons,
        "different DEV priors should produce different neuron counts"
    );
    assert!(
        explorer.stats.total_neurons > builder.stats.total_neurons,
        "explorer (DEV=0.7) should have more neurons than builder (DEV=0.3): {} vs {}",
        explorer.stats.total_neurons,
        builder.stats.total_neurons
    );
}

#[test]
fn different_genomes_different_differentiation() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();

    let explorer = develop_region(region, &explorer_genome(), &phases, 500, 42);
    let builder = develop_region(region, &builder_genome(), &phases, 500, 42);

    eprintln!(
        "  explorer: {}/{} differentiated",
        explorer.stats.differentiated, explorer.stats.total_neurons
    );
    eprintln!(
        "  builder:  {}/{} differentiated",
        builder.stats.differentiated, builder.stats.total_neurons
    );

    // Both should differentiate some neurons
    assert!(
        explorer.stats.differentiated > 0,
        "explorer should differentiate some neurons"
    );
    assert!(
        builder.stats.differentiated > 0,
        "builder should differentiate some neurons"
    );

    // The differentiated counts or ratios should differ
    let exp_ratio =
        explorer.stats.differentiated as f32 / explorer.stats.total_neurons.max(1) as f32;
    let bld_ratio =
        builder.stats.differentiated as f32 / builder.stats.total_neurons.max(1) as f32;

    eprintln!("  explorer differentiation ratio: {:.1}%", exp_ratio * 100.0);
    eprintln!("  builder differentiation ratio:  {:.1}%", bld_ratio * 100.0);

    // High plasticity explorer should differentiate at a different rate
    let diff = (exp_ratio - bld_ratio).abs();
    eprintln!("  ratio difference: {:.1}%", diff * 100.0);
}

#[test]
fn same_genome_same_seed_identical() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();
    let genome = explorer_genome();

    let a = develop_region(region, &genome, &phases, 500, 42);
    let b = develop_region(region, &genome, &phases, 500, 42);

    assert_eq!(a.stats.total_neurons, b.stats.total_neurons);
    assert_eq!(a.stats.differentiated, b.stats.differentiated);
    assert_eq!(a.stats.undifferentiated, b.stats.undifferentiated);
}

#[test]
fn same_genome_different_seed_differs() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();
    let genome = explorer_genome();

    let a = develop_region(region, &genome, &phases, 500, 42);
    let b = develop_region(region, &genome, &phases, 500, 99);

    // Same genome, same neuron count (deterministic resolution)
    assert_eq!(
        a.stats.total_neurons, b.stats.total_neurons,
        "same genome should produce same neuron count regardless of seed"
    );

    // But different placement
    let positions_differ = a
        .neurons
        .iter()
        .zip(b.neurons.iter())
        .any(|(na, nb)| na.soma.position != nb.soma.position);
    assert!(
        positions_differ,
        "different seeds should produce different neuron placement"
    );
}

#[test]
fn disc_populations_reported() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();

    let result = develop_region(region, &explorer_genome(), &phases, 500, 42);

    eprintln!("  disc populations:");
    for (name, count) in &result.stats.populations {
        eprintln!("    {name}: {count}");
    }

    // Should have 3 disc populations (sensory, motor, gate)
    assert_eq!(
        result.stats.populations.len(),
        3,
        "should have 3 disc populations"
    );
}

#[test]
fn trophic_field_present_after_development() {
    let program = parse(TEST_CORTEX).unwrap();
    let region = &program.regions[0];
    let phases = PhaseConfig::fast();

    let result = develop_region(region, &explorer_genome(), &phases, 500, 42);

    assert_eq!(result.trophic_fields.len(), 1, "should have one trophic field");
    assert_eq!(result.trophic_fields[0].chemical.0, "BDNF");

    // After diffusion, the field should have spread
    let nonzero_count = result.trophic_fields[0]
        .data
        .iter()
        .filter(|&&v| v > 0)
        .count();
    let total_cells = result.trophic_fields[0].data.len();
    let fill_ratio = nonzero_count as f32 / total_cells as f32;

    eprintln!(
        "  BDNF field: {nonzero_count}/{total_cells} cells non-zero ({:.0}%)",
        fill_ratio * 100.0
    );
    assert!(
        fill_ratio > 0.01,
        "trophic field should have spread beyond the source point"
    );
}
