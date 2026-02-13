//! # astromind-final
//!
//! Developmental growth language for the Astromind spatial neuronal system.
//!
//! `.neuro` files describe species templates — developmental programs that,
//! when executed with a genome and metabolic budget, produce spatial neuron
//! pools through trophic-factor-driven differentiation.
//!
//! Two constructs:
//! - `region` — brain tissue (grows, differentiates, wires)
//! - `attractor` — peripheral tissue (emits trophic factors inward)

pub mod development;
pub mod error;
pub mod parser;
pub mod resolve;
pub mod trophic;
pub mod validator;

pub use error::NeuroError;
pub use parser::ast::NeuroProgram;
pub use parser::parse;
