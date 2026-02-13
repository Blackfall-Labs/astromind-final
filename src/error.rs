use std::fmt;

/// All errors produced by astromind-final.
#[derive(Debug)]
pub enum NeuroError {
    /// Lexer or parser error with source location.
    Parse(ParseError),
    /// Validation error (cycles, missing chemicals, budget infeasibility).
    Validation(ValidationError),
}

/// A parse error with source location and message.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
    pub col: u32,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ParseError {}

/// A validation diagnostic.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
}

/// Validation error — contains one or more diagnostics.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub diagnostics: Vec<Diagnostic>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for d in &self.diagnostics {
            writeln!(f, "[{:?}] {}", d.level, d.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

impl fmt::Display for NeuroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuroError::Parse(e) => write!(f, "parse error: {e}"),
            NeuroError::Validation(e) => write!(f, "validation error: {e}"),
        }
    }
}

impl std::error::Error for NeuroError {}

impl From<ParseError> for NeuroError {
    fn from(e: ParseError) -> Self {
        NeuroError::Parse(e)
    }
}

impl From<ValidationError> for NeuroError {
    fn from(e: ValidationError) -> Self {
        NeuroError::Validation(e)
    }
}
