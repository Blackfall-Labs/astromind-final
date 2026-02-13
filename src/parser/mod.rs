//! Parser for .neuro developmental programs.
//!
//! Hand-written recursive descent parser: source text -> tokens -> AST.

pub mod ast;
pub mod grammar;
pub mod lexer;

use crate::error::ParseError;
use ast::NeuroProgram;
use grammar::Parser;

/// Parse a .neuro source string into a NeuroProgram AST.
pub fn parse(source: &str) -> Result<NeuroProgram, ParseError> {
    let mut parser = Parser::new(source)?;
    parser.parse_program()
}
