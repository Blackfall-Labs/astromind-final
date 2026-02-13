//! Population budget feasibility diagnostics.
//!
//! Checks for issues that won't hard-fail but indicate design problems:
//! - Multiple `:dominant` entries in the same populate block (they'll compete)
//! - Empty populate blocks
//! - Differentiation blocks targeting types not in the populate mix

use std::collections::HashSet;

use crate::error::{Diagnostic, DiagnosticLevel};
use crate::parser::ast::NeuroProgram;

/// Known intensity levels from lowest to highest priority.
const DOMINANT_SYMBOL: &str = "dominant";

/// Check population budget feasibility.
pub fn check_budget(program: &NeuroProgram) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    for region in &program.regions {
        let rname = region.name.as_str();

        if let Some(populate) = &region.populate {
            // Check for empty populate
            if populate.entries.is_empty() {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!("region :{rname}: populate block is empty"),
                    region: Some(rname.to_string()),
                });
                continue;
            }

            // Check for multiple dominant entries
            let dominant_count = populate
                .entries
                .iter()
                .filter(|e| e.intensity.as_str() == DOMINANT_SYMBOL)
                .count();

            if dominant_count > 1 {
                let dominant_types: Vec<_> = populate
                    .entries
                    .iter()
                    .filter(|e| e.intensity.as_str() == DOMINANT_SYMBOL)
                    .map(|e| format!("{}", e.cell_type))
                    .collect();
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!(
                        "region :{rname}: {} cell types claim :dominant priority [{}] — \
                         they will compete aggressively for metabolic budget",
                        dominant_count,
                        dominant_types.join(", ")
                    ),
                    region: Some(rname.to_string()),
                });
            }

            // Collect populated cell types
            let populated_types: HashSet<&str> =
                populate.entries.iter().map(|e| e.cell_type.as_str()).collect();

            // Check differentiation blocks reference types in the populate mix
            for diff in &region.differentiations {
                let target = diff.target_type.as_str();
                if !populated_types.contains(target) {
                    diagnostics.push(Diagnostic {
                        level: DiagnosticLevel::Warning,
                        message: format!(
                            "region :{rname}: differentiate {} targets a cell type \
                             not present in the populate block",
                            diff.target_type
                        ),
                        region: Some(rname.to_string()),
                    });
                }
            }
        } else if !region.differentiations.is_empty() {
            // Has differentiation but no populate
            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Warning,
                message: format!(
                    "region :{rname}: has differentiation blocks but no populate block — \
                     no neurons to differentiate"
                ),
                region: Some(rname.to_string()),
            });
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

    fn entry(cell_type: &str, intensity: &str) -> PopulateEntry {
        PopulateEntry {
            cell_type: Symbol(cell_type.to_string()),
            intensity: Symbol(intensity.to_string()),
            span: Span::default(),
        }
    }

    #[test]
    fn valid_populate_no_warnings() {
        let mut r = region("cortex");
        r.populate = Some(PopulateBlock {
            entries: vec![
                entry("pyramidal", "dominant"),
                entry("inhibitory", "moderate"),
                entry("gate", "sparse"),
            ],
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_budget(&program);
        assert!(diags.is_empty(), "valid populate should have no warnings: {diags:?}");
    }

    #[test]
    fn multiple_dominant_warns() {
        let mut r = region("cortex");
        r.populate = Some(PopulateBlock {
            entries: vec![
                entry("pyramidal", "dominant"),
                entry("inhibitory", "dominant"),
                entry("gate", "sparse"),
            ],
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_budget(&program);
        assert!(
            diags.iter().any(|d| d.message.contains("dominant")),
            "should warn about multiple dominant: {diags:?}"
        );
    }

    #[test]
    fn diff_without_populate_warns() {
        let mut r = region("test");
        r.differentiations.push(DifferentiateBlock {
            target_type: Symbol("sensory".to_string()),
            span: Span::default(),
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
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
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_budget(&program);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("no populate block")),
            "should warn about missing populate: {diags:?}"
        );
    }

    #[test]
    fn diff_targeting_missing_type_warns() {
        let mut r = region("cortex");
        r.populate = Some(PopulateBlock {
            entries: vec![entry("pyramidal", "dominant"), entry("inhibitory", "moderate")],
            span: Span::default(),
        });
        r.differentiations.push(DifferentiateBlock {
            target_type: Symbol("modulator".to_string()),
            span: Span::default(),
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
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
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_budget(&program);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("not present in the populate block")),
            "should warn about missing type: {diags:?}"
        );
    }

    #[test]
    fn empty_populate_warns() {
        let mut r = region("test");
        r.populate = Some(PopulateBlock {
            entries: Vec::new(),
            span: Span::default(),
        });

        let program = NeuroProgram {
            regions: vec![r],
            attractors: Vec::new(),
        };
        let diags = check_budget(&program);
        assert!(
            diags.iter().any(|d| d.message.contains("empty")),
            "should warn about empty populate: {diags:?}"
        );
    }
}
