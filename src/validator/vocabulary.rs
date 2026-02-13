//! Trophic chemical vocabulary validation.
//!
//! Every chemical referenced in a .neuro program must have a source:
//! 1. Secreted by a region (`secretes :chemical`)
//! 2. Emitted by an attractor (`emits :chemical, :intensity`)
//! 3. Defined as a gradient chemical within a region
//! 4. A built-in chemical (DA, GABA, ACh, BDNF, Wnt, BMP, FGF, EphrinA, NGF)

use std::collections::HashSet;

use crate::error::{Diagnostic, DiagnosticLevel};
use crate::parser::ast::NeuroProgram;
use crate::trophic::chemical::TrophicVocabulary;

/// Check that all referenced chemicals are sourced.
pub fn check_vocabulary(program: &NeuroProgram) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Collect all sourced chemicals
    let mut sourced: HashSet<String> = HashSet::new();

    // Built-in chemicals are always available
    let vocab = TrophicVocabulary::new();
    for name in [
        "DA", "GABA", "ACh", "BDNF", "Wnt", "BMP", "FGF", "EphrinA", "NGF",
    ] {
        if vocab.contains(name) {
            sourced.insert(name.to_string());
        }
    }

    // Region secretions
    for region in &program.regions {
        for chem in &region.secretes {
            sourced.insert(chem.as_str().to_string());
        }
        // Gradient chemicals defined within the region are also sourced locally
        for gradient in &region.gradients {
            if let Some(chem) = &gradient.chemical {
                sourced.insert(chem.as_str().to_string());
            }
        }
    }

    // Attractor emissions
    for attractor in &program.attractors {
        for emit in &attractor.emits {
            sourced.insert(emit.chemical.as_str().to_string());
        }
    }

    // Now check all references
    for region in &program.regions {
        let rname = region.name.as_str();

        // grows toward/facing :chemical
        for grows in &region.grows {
            if let Some(target) = &grows.target {
                if !sourced.contains(target.as_str()) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Error,
                        message: format!(
                            "region :{rname} grows toward {target}, \
                             but no region secretes it and no attractor emits it"
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }
        }

        // differentiate where :chemical
        for diff in &region.differentiations {
            for wc in &diff.where_clauses {
                if !sourced.contains(wc.chemical.as_str()) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Error,
                        message: format!(
                            "region :{rname}, differentiate {}: \
                             where clause references unsourced chemical {}",
                            diff.target_type, wc.chemical
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }

            // receptive_to
            for chem in &diff.receptive_to {
                if !sourced.contains(chem.as_str()) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Error,
                        message: format!(
                            "region :{rname}, differentiate {}: \
                             receptive_to references unsourced chemical {chem}",
                            diff.target_type
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }

            // sensitive_to
            for chem in &diff.sensitive_to {
                if !sourced.contains(chem.as_str()) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Warning,
                        message: format!(
                            "region :{rname}, differentiate {}: \
                             sensitive_to references unsourced chemical {chem}",
                            diff.target_type
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }
        }

        // wire within :chemical
        for wire in &region.wiring {
            for zone in &wire.zone_overrides {
                if !sourced.contains(zone.chemical.as_str()) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Error,
                        message: format!(
                            "region :{rname}, wire within {}: \
                             references unsourced chemical",
                            zone.chemical
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }
        }

        // plasticity_from
        for chem in &region.plasticity_from {
            if !sourced.contains(chem.as_str()) {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!(
                        "region :{rname}: plasticity_from references unsourced chemical {chem}"
                    ),
                    region: Some(rname.to_string()),
                });
            }
        }

        // consolidation_from
        for chem in &region.consolidation_from {
            if !sourced.contains(chem.as_str()) {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!(
                        "region :{rname}: consolidation_from references unsourced chemical {chem}"
                    ),
                    region: Some(rname.to_string()),
                });
            }
        }

        // sensitive_to (region-level)
        for chem in &region.sensitive_to {
            if !sourced.contains(chem.as_str()) {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!(
                        "region :{rname}: sensitive_to references unsourced chemical {chem}"
                    ),
                    region: Some(rname.to_string()),
                });
            }
        }
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
    fn builtin_chemicals_always_valid() {
        let mut r = region("test");
        r.differentiations.push(DifferentiateBlock {
            target_type: Symbol("sensory".to_string()),
            span: Span::default(),
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
            between: None,
            where_clauses: vec![WhereClause {
                chemical: Symbol("BDNF".to_string()),
                level: Symbol("high".to_string()),
            }],
            where_dense: false,
            profile: None,
            inhibitory: None,
            recurrent: None,
            feedforward: None,
            sparse: None,
            period: None,
            bank: None,
            sensitive_to: Vec::new(),
        });
        r.grows.push(GrowsDirective {
            mode: GrowsMode::First,
            target: None,
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_vocabulary(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "built-in chemical should be valid: {diags:?}"
        );
    }

    #[test]
    fn secreted_chemical_is_valid() {
        let mut bs = region("brainstem");
        bs.secretes.push(Symbol("brainstem_signal".to_string()));

        let mut ctx = region("cortex");
        ctx.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("brainstem_signal".to_string())),
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![bs, ctx],
            attractors: Vec::new(),
        };
        let diags = check_vocabulary(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "secreted chemical should be valid: {diags:?}"
        );
    }

    #[test]
    fn unsourced_grows_target_is_error() {
        let mut r = region("cortex");
        r.grows.push(GrowsDirective {
            mode: GrowsMode::Toward,
            target: Some(Symbol("nonexistent_factor".to_string())),
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_vocabulary(&program);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(
            !errors.is_empty(),
            "unsourced chemical should produce error"
        );
        assert!(errors[0].message.contains("nonexistent_factor"));
    }

    #[test]
    fn attractor_emission_satisfies_reference() {
        let mut ctx = region("cortex");
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
        let diags = check_vocabulary(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "attractor emission should satisfy reference: {diags:?}"
        );
    }

    #[test]
    fn unsourced_where_clause_is_error() {
        let mut r = region("test");
        r.differentiations.push(DifferentiateBlock {
            target_type: Symbol("motor".to_string()),
            span: Span::default(),
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
            between: None,
            where_clauses: vec![WhereClause {
                chemical: Symbol("phantom_chemical".to_string()),
                level: Symbol("high".to_string()),
            }],
            where_dense: false,
            profile: None,
            inhibitory: None,
            recurrent: None,
            feedforward: None,
            sparse: None,
            period: None,
            bank: None,
            sensitive_to: Vec::new(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_vocabulary(&program);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(
            !errors.is_empty(),
            "unsourced where chemical should be error"
        );
        assert!(errors[0].message.contains("phantom_chemical"));
    }

    #[test]
    fn gradient_chemical_is_self_sourced() {
        let mut r = region("hippocampus");
        r.gradients.push(GradientBlock {
            name: Symbol("ca_gradient".to_string()),
            span: Span::default(),
            chemical: Some(Symbol("local_factor".to_string())),
            source: None,
            diffusion: None,
        });
        r.differentiations.push(DifferentiateBlock {
            target_type: Symbol("pyramidal".to_string()),
            span: Span::default(),
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
            between: None,
            where_clauses: vec![WhereClause {
                chemical: Symbol("local_factor".to_string()),
                level: Symbol("high".to_string()),
            }],
            where_dense: false,
            profile: None,
            inhibitory: None,
            recurrent: None,
            feedforward: None,
            sparse: None,
            period: None,
            bank: None,
            sensitive_to: Vec::new(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_vocabulary(&program);
        assert!(
            diags.iter().all(|d| d.level != DiagnosticLevel::Error),
            "gradient chemical should be self-sourced: {diags:?}"
        );
    }
}
