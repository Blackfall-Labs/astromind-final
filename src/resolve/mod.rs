//! Symbol resolution — maps .neuro intensity symbols to concrete values
//! through genome modulation and metabolic budget competition.

pub mod budget;
pub mod genome;
pub mod intensity;

pub use budget::MetabolicBudget;
pub use genome::GenomeProfile;
pub use intensity::{IntensityLevel, ResolvedPopulation};
