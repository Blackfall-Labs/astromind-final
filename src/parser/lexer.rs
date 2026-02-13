//! Hand-written lexer for .neuro files.
//!
//! Tokens: keywords, symbols (:name), identifiers, integers, bools, comma, newline.

use crate::error::ParseError;
use crate::parser::ast::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Region,
    Attractor,
    Do,
    End,
    Populate,
    Differentiate,
    Wire,
    Gradient,
    Within,
    Body,

    // Literals
    Ident(String),
    Symbol(String), // without leading colon
    Integer(u32),
    Bool(bool),

    // Punctuation
    Comma,
    Newline,

    Eof,
}

impl Token {
    pub fn is_newline(&self) -> bool {
        matches!(self, Token::Newline)
    }
}

pub struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            source: source.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.source.get(self.pos).copied()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn skip_whitespace_not_newline(&mut self) {
        while let Some(b) = self.peek_byte() {
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        // # comment until end of line
        while let Some(b) = self.peek_byte() {
            if b == b'\n' {
                break;
            }
            self.advance();
        }
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn read_identifier(&mut self) -> String {
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.source[start..self.pos]).into_owned()
    }

    fn read_integer(&mut self) -> Result<u32, ParseError> {
        let span = self.span();
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        let text = String::from_utf8_lossy(&self.source[start..self.pos]);
        text.parse::<u32>().map_err(|_| ParseError {
            message: format!("invalid integer: {text}"),
            line: span.line,
            col: span.col,
        })
    }

    fn classify_keyword(word: &str) -> Token {
        match word {
            "region" => Token::Region,
            "attractor" => Token::Attractor,
            "do" => Token::Do,
            "end" => Token::End,
            "populate" => Token::Populate,
            "differentiate" => Token::Differentiate,
            "wire" => Token::Wire,
            "gradient" => Token::Gradient,
            "within" => Token::Within,
            "body" => Token::Body,
            "true" => Token::Bool(true),
            "false" => Token::Bool(false),
            _ => Token::Ident(word.to_string()),
        }
    }

    /// Produce the next token.
    pub fn next_token(&mut self) -> Result<(Token, Span), ParseError> {
        self.skip_whitespace_not_newline();

        let span = self.span();

        match self.peek_byte() {
            None => Ok((Token::Eof, span)),

            Some(b'#') => {
                self.skip_comment();
                // After comment, return newline if at newline, else recurse
                if self.peek_byte() == Some(b'\n') {
                    self.advance();
                    Ok((Token::Newline, span))
                } else {
                    // EOF after comment
                    Ok((Token::Eof, span))
                }
            }

            Some(b'\n') => {
                self.advance();
                Ok((Token::Newline, span))
            }

            Some(b',') => {
                self.advance();
                Ok((Token::Comma, span))
            }

            Some(b':') => {
                self.advance(); // consume colon
                let first = self.peek_byte();
                if first.map_or(false, |b| b.is_ascii_alphabetic() || b == b'_') {
                    let name = self.read_identifier();
                    Ok((Token::Symbol(name), span))
                } else {
                    Err(ParseError {
                        message: "expected identifier after ':'".into(),
                        line: span.line,
                        col: span.col,
                    })
                }
            }

            Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                let word = self.read_identifier();
                Ok((Self::classify_keyword(&word), span))
            }

            Some(b) if b.is_ascii_digit() => {
                let val = self.read_integer()?;
                Ok((Token::Integer(val), span))
            }

            Some(b) => {
                self.advance();
                Err(ParseError {
                    message: format!("unexpected character: '{}'", b as char),
                    line: span.line,
                    col: span.col,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(src: &str) -> Vec<Token> {
        let mut lex = Lexer::new(src);
        let mut out = Vec::new();
        loop {
            let (tok, _) = lex.next_token().unwrap();
            if tok == Token::Eof {
                break;
            }
            out.push(tok);
        }
        out
    }

    #[test]
    fn symbol_literal() {
        assert_eq!(tokens(":foo"), vec![Token::Symbol("foo".into())]);
        assert_eq!(
            tokens(":m1_cortex"),
            vec![Token::Symbol("m1_cortex".into())]
        );
    }

    #[test]
    fn keywords() {
        let toks = tokens("region do end");
        assert_eq!(toks, vec![Token::Region, Token::Do, Token::End]);
    }

    #[test]
    fn identifiers() {
        let toks = tokens("pyramidal inhibitory");
        assert_eq!(
            toks,
            vec![
                Token::Ident("pyramidal".into()),
                Token::Ident("inhibitory".into()),
            ]
        );
    }

    #[test]
    fn booleans() {
        let toks = tokens("true false");
        assert_eq!(toks, vec![Token::Bool(true), Token::Bool(false)]);
    }

    #[test]
    fn comment_skipped() {
        let toks = tokens("region # this is a comment\nend");
        assert_eq!(toks, vec![Token::Region, Token::Newline, Token::End]);
    }

    #[test]
    fn comma() {
        let toks = tokens(":a, :b");
        assert_eq!(
            toks,
            vec![
                Token::Symbol("a".into()),
                Token::Comma,
                Token::Symbol("b".into()),
            ]
        );
    }

    #[test]
    fn newlines_preserved() {
        let toks = tokens("region\n\nend");
        assert_eq!(
            toks,
            vec![Token::Region, Token::Newline, Token::Newline, Token::End]
        );
    }

    #[test]
    fn integer_literal() {
        let toks = tokens("42");
        assert_eq!(toks, vec![Token::Integer(42)]);
    }

    #[test]
    fn full_region_header() {
        let toks = tokens("region :m1_cortex do");
        assert_eq!(
            toks,
            vec![
                Token::Region,
                Token::Symbol("m1_cortex".into()),
                Token::Do,
            ]
        );
    }

    #[test]
    fn attractor_with_emits() {
        let toks = tokens("attractor :keyboard do\n  emits :language_trophic, :dominant\nend");
        // Flatten and check key tokens present
        assert_eq!(toks[0], Token::Attractor);
        assert_eq!(toks[1], Token::Symbol("keyboard".into()));
        assert_eq!(toks[2], Token::Do);
    }
}
