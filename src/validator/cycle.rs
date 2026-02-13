//! Trophic dependency DAG cycle detection.
//!
//! Builds a directed graph: region A depends on region B if A `grows toward :X`
//! and B `secretes :X`. The seed region (`grows first`) must have no incoming
//! edges. Cycles are rejected — they would cause deadlocked development.

use std::collections::{HashMap, HashSet};

use crate::error::{Diagnostic, DiagnosticLevel};
use crate::parser::ast::{GrowsMode, NeuroProgram};

/// Check for cycles in the trophic dependency graph.
pub fn check_cycles(program: &NeuroProgram) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Build: chemical -> set of regions that secrete it
    let mut secreted_by: HashMap<&str, Vec<&str>> = HashMap::new();
    for region in &program.regions {
        for chem in &region.secretes {
            secreted_by
                .entry(chem.as_str())
                .or_default()
                .push(region.name.as_str());
        }
    }
    // Attractors also emit chemicals (they are sources, not dependents)
    for attractor in &program.attractors {
        for emit in &attractor.emits {
            secreted_by
                .entry(emit.chemical.as_str())
                .or_default()
                .push(attractor.name.as_str());
        }
    }

    // Build dependency graph: region -> set of regions it depends on
    let mut depends_on: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut seed_regions: Vec<&str> = Vec::new();

    for region in &program.regions {
        let name = region.name.as_str();

        for grows in &region.grows {
            match grows.mode {
                GrowsMode::First => {
                    seed_regions.push(name);
                }
                GrowsMode::Toward | GrowsMode::Facing => {
                    if let Some(target) = &grows.target {
                        // This region depends on whoever secretes/emits this chemical
                        if let Some(sources) = secreted_by.get(target.as_str()) {
                            for &source in sources {
                                if source != name {
                                    depends_on.entry(name).or_default().insert(source);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Check: at least one seed region
    if seed_regions.is_empty() && !program.regions.is_empty() {
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Error,
            message: "no seed region found — at least one region must declare `grows first`"
                .to_string(),
            region: None,
        });
    }

    // Check: seed regions must have no incoming dependencies on other regions
    // (dependencies on attractors are fine — attractors are external sources)
    let region_names: HashSet<&str> = program.regions.iter().map(|r| r.name.as_str()).collect();
    for &seed in &seed_regions {
        if let Some(deps) = depends_on.get(seed) {
            let region_deps: Vec<_> = deps
                .iter()
                .filter(|&&d| region_names.contains(d))
                .copied()
                .collect();
            if !region_deps.is_empty() {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Error,
                    message: format!(
                        "seed region :{seed} has trophic dependencies on [{}] — \
                         seed regions must have no prerequisites from other regions",
                        region_deps.join(", :")
                    ),
                    region: Some(seed.to_string()),
                });
            }
        }
    }

    // Topological sort to detect cycles (Kahn's algorithm)
    let all_regions: Vec<&str> = program.regions.iter().map(|r| r.name.as_str()).collect();

    // Compute in-degree for each region
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for &r in &all_regions {
        in_degree.insert(r, 0);
    }
    for (node, deps) in &depends_on {
        // Only count edges where both ends are regions (not attractors)
        for &dep in deps {
            if in_degree.contains_key(dep) {
                // dep -> node edge exists in the dependency graph
                // node depends on dep, so node has an incoming edge
                *in_degree.entry(node).or_default() += 1;
            }
        }
    }

    // Start with nodes that have no incoming edges
    let mut queue: Vec<&str> = all_regions
        .iter()
        .filter(|&&r| in_degree.get(r).copied().unwrap_or(0) == 0)
        .copied()
        .collect();

    let mut visited = 0usize;

    // Build reverse adjacency: dep -> vec of nodes that depend on it
    let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();
    for (&node, deps) in &depends_on {
        for &dep in deps {
            if all_regions.contains(&dep) {
                dependents.entry(dep).or_default().push(node);
            }
        }
    }

    while let Some(node) = queue.pop() {
        visited += 1;
        if let Some(downstream) = dependents.get(node) {
            for &d in downstream {
                if let Some(deg) = in_degree.get_mut(d) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(d);
                    }
                }
            }
        }
    }

    if visited < all_regions.len() {
        // Find the regions that are part of cycles
        let cycle_members: Vec<&str> = all_regions
            .iter()
            .filter(|&&r| in_degree.get(r).copied().unwrap_or(0) > 0)
            .copied()
            .collect();

        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Error,
            message: format!(
                "trophic dependency cycle detected among regions: [{}]",
                cycle_members
                    .iter()
                    .map(|r| format!(":{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            region: None,
        });
    }

    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::*;

    fn region(name: &str) -> RegionDef {
        RegionDef {
            name: Symbol(name.to_string()),
            span: Span::default(),
            shape: None,
            proportion: None,
            grows: Vec::new(),
            populate: None,
            differentiations: Vec::new(),
            wiring: Vec::new(),
            gradients: Vec::new(),
            secretes: Vec::new(),
            plasticity_from: Vec::new(),
            consolidation_from: Vec::new(),
            sensitive_to: Vec::new(),
        }
    }

    #[test]
    fn empty_program_no_errors() {
        let program = NeuroProgram {
            regions: Vec::new(),
            attractors: Vec::new(),
        };
        let diags = check_cycles(&program);
        assert!(diags.is_empty());
    }

    #[test]
    fn single_seed_region_ok() {
        let mut r = region("brainstem");
        r.grows.push(GrowsDirective {
            mode: GrowsMode::First,
            target: None,
            span: Span::default(),
        });
        r.secretes.push(Symbol("brainstem_signal".to_string()));

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_cycles(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "single seed should produce no errors: {diags:?}"
        );
    }

    #[test]
    fn linear_chain_ok() {
        // brainstem (seed) -> secretes :bs_signal
        // cortex -> grows toward :bs_signal, secretes :cortex_signal
        let mut bs = region("brainstem");
        bs.grows.push(GrowsDirective {
            mode: GrowsMode::First,
            target: None,
            span: Span::default(),
        });
        bs.secretes.push(Symbol("bs_signal".to_string()));

        let mut ctx = region("cortex");
        ctx.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("bs_signal".to_string())),
            span: Span::default(),
        });
        ctx.secretes.push(Symbol("cortex_signal".to_string()));

        let program = NeuroProgram {
            regions: vec![bs, ctx],
            attractors: Vec::new(),
        };
        let diags = check_cycles(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "linear chain should produce no errors: {diags:?}"
        );
    }

    #[test]
    fn mutual_dependency_is_cycle() {
        // A grows toward :b_signal, secretes :a_signal
        // B grows toward :a_signal, secretes :b_signal
        // No seed — also an error
        let mut a = region("a");
        a.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("b_signal".to_string())),
            span: Span::default(),
        });
        a.secretes.push(Symbol("a_signal".to_string()));

        let mut b = region("b");
        b.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("a_signal".to_string())),
            span: Span::default(),
        });
        b.secretes.push(Symbol("b_signal".to_string()));

        let program = NeuroProgram {
            regions: vec![a, b],
            attractors: Vec::new(),
        };
        let diags = check_cycles(&program);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(!errors.is_empty(), "mutual dependency should produce errors");
        assert!(
            errors.iter().any(|d| d.message.contains("cycle")),
            "should report cycle: {errors:?}"
        );
    }

    #[test]
    fn no_seed_is_error() {
        let mut r = region("cortex");
        r.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("something".to_string())),
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_cycles(&program);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(
            errors.iter().any(|d| d.message.contains("seed")),
            "should report missing seed: {errors:?}"
        );
    }

    #[test]
    fn attractor_as_source_breaks_dependency() {
        // body emits :hand_signal
        // cortex grows toward :hand_signal (depends on attractor, not a cycle)
        let mut ctx = region("cortex");
        ctx.grows.push(GrowsDirective {
            mode: GrowsMode::First,
            target: None,
            span: Span::default(),
        });
        ctx.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("hand_signal".to_string())),
            span: Span::default(),
        });

        let attractor = AttractorDef {
            name: Symbol("hand".to_string()),
            span: Span::default(),
            emits: vec![EmitsDirective {
                chemical: Symbol("hand_signal".to_string()),
                intensity: Symbol("moderate".to_string()),
                span: Span::default(),
            }],
        };

        let program = NeuroProgram {
            regions: vec![ctx],
            attractors: vec![attractor],
        };
        let diags = check_cycles(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "attractor source should not create cycle: {diags:?}"
        );
    }
}
