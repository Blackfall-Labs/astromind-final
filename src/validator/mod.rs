//! Validation passes for .neuro programs.
//!
//! Three checks:
//! 1. **Cycle detection** — trophic dependency DAG must be acyclic
//! 2. **Vocabulary** — all referenced chemicals must be sourced
//! 3. **Budget** — population feasibility diagnostics

pub mod budget;
pub mod cycle;
pub mod vocabulary;

use crate::error::{Diagnostic, DiagnosticLevel, ValidationError};
use crate::parser::ast::NeuroProgram;

/// Run all validation passes on a parsed program.
/// Returns Ok(diagnostics) with warnings, or Err if any errors found.
pub fn validate(program: &NeuroProgram) -> Result<Vec<Diagnostic>, ValidationError> {
    let mut diagnostics = Vec::new();

    diagnostics.extend(cycle::check_cycles(program));
    diagnostics.extend(vocabulary::check_vocabulary(program));
    diagnostics.extend(budget::check_budget(program));

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == DiagnosticLevel::Error);

    if has_errors {
        Err(ValidationError { diagnostics })
    } else {
        Ok(diagnostics)
    }
}
