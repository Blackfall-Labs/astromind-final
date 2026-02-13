//! AST types for .neuro developmental programs.
//!
//! A NeuroProgram contains region definitions (brain tissue) and
//! attractor definitions (peripheral tissue / body parts).

/// A complete .neuro program — one or more regions and/or attractors.
#[derive(Debug, Clone)]
pub struct NeuroProgram {
    pub regions: Vec<RegionDef>,
    pub attractors: Vec<AttractorDef>,
}

/// Symbol literal (e.g., :m1_cortex, :high, :DA).
/// Stored without the leading colon.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ":{}", self.0)
    }
}

/// Source location for error reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

// ─── Region ─────────────────────────────────────────────────────────────────

/// A brain tissue region definition.
#[derive(Debug, Clone)]
pub struct RegionDef {
    pub name: Symbol,
    pub span: Span,
    pub shape: Option<Symbol>,
    pub proportion: Option<Symbol>,
    pub grows: Vec<GrowsDirective>,
    pub populate: Option<PopulateBlock>,
    pub differentiations: Vec<DifferentiateBlock>,
    pub wiring: Vec<WireBlock>,
    pub gradients: Vec<GradientBlock>,
    pub secretes: Vec<Symbol>,
    pub plasticity_from: Vec<Symbol>,
    pub consolidation_from: Vec<Symbol>,
    pub sensitive_to: Vec<Symbol>,
}

/// How a region grows.
#[derive(Debug, Clone)]
pub struct GrowsDirective {
    pub mode: GrowsMode,
    /// Target symbol — None for `grows first`.
    pub target: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowsMode {
    /// grows toward :chemical — grow toward trophic factor source
    Toward,
    /// grows facing :chemical — orient output face toward demand
    Facing,
    /// grows first — the seed region, no prerequisites
    First,
}

// ─── Populate ───────────────────────────────────────────────────────────────

/// Population mix within a region.
#[derive(Debug, Clone)]
pub struct PopulateBlock {
    pub entries: Vec<PopulateEntry>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct PopulateEntry {
    pub cell_type: Symbol,
    pub intensity: Symbol,
    pub span: Span,
}

// ─── Differentiate ──────────────────────────────────────────────────────────

/// A differentiation rule — conditions under which neurons commit to a type.
#[derive(Debug, Clone)]
pub struct DifferentiateBlock {
    pub target_type: Symbol,
    pub span: Span,
    pub receptive_to: Vec<Symbol>,
    pub affinity: Option<Symbol>,
    pub abundance: Option<Symbol>,
    pub between: Option<(Symbol, Symbol)>,
    pub where_clauses: Vec<WhereClause>,
    pub where_dense: bool,
    pub profile: Option<Symbol>,
    pub inhibitory: Option<bool>,
    pub recurrent: Option<bool>,
    pub feedforward: Option<bool>,
    pub sparse: Option<bool>,
    pub period: Option<Symbol>,
    pub bank: Option<Symbol>,
    pub sensitive_to: Vec<Symbol>,
}

/// A chemical-zone condition: `where :chemical, :level`
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub chemical: Symbol,
    pub level: Symbol,
}

// ─── Wire ───────────────────────────────────────────────────────────────────

/// Wiring configuration for a region.
#[derive(Debug, Clone)]
pub struct WireBlock {
    pub mode: Symbol,
    pub span: Span,
    pub global_params: WireParams,
    pub zone_overrides: Vec<WireZoneOverride>,
}

#[derive(Debug, Clone, Default)]
pub struct WireParams {
    pub reach: Option<Symbol>,
    pub fanout: Option<Symbol>,
    pub strength: Option<Symbol>,
}

/// Zone-specific wiring override: `within :chemical, :level do ... end`
#[derive(Debug, Clone)]
pub struct WireZoneOverride {
    pub chemical: Symbol,
    pub level: Symbol,
    pub params: WireParams,
}

// ─── Gradient ───────────────────────────────────────────────────────────────

/// Internal gradient definition within a region.
#[derive(Debug, Clone)]
pub struct GradientBlock {
    pub name: Symbol,
    pub span: Span,
    pub chemical: Option<Symbol>,
    pub source: Option<Symbol>,
    pub diffusion: Option<Symbol>,
}

// ─── Attractor ──────────────────────────────────────────────────────────────

/// A peripheral attractor (body part) that emits trophic factors.
#[derive(Debug, Clone)]
pub struct AttractorDef {
    pub name: Symbol,
    pub span: Span,
    pub emits: Vec<EmitsDirective>,
}

/// A trophic emission: `emits :chemical, :intensity`
#[derive(Debug, Clone)]
pub struct EmitsDirective {
    pub chemical: Symbol,
    pub intensity: Symbol,
    pub span: Span,
}
